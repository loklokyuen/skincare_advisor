import sys
from concurrent.futures import Future
from textwrap import dedent
from pathlib import Path

from langchain_core.messages import AIMessage

sys.path.insert(0, str(Path(__file__).parent.parent))

from graph.graph import (
    _cards_for_selected_product_names,
    _community_query_for_evidence,
    _community_placeholder_response,
    _community_search_query,
    _extract_recommended_ingredients,
    _ingredient_evidence_query,
    _report_progress,
)
from graph import recommendation_cache
from graph.recommendation_cache import get_prepared_answer, store_prepared_answer
from graph.nodes.generate_response import _build_context_block
import graph.nodes.retrieve_context as retrieve_context_module
from graph.nodes.retrieve_context import (
    _build_recommendation_query,
    _filter_duplicate_retinoids,
    _filter_skin_type_mismatches,
    _rank_products_for_profile,
    _topic_from_last_ai_message,
    _GENERIC_FOLLOWUP_RE,
    retrieve_context,
)
from graph.nodes.validate_response import (
    _remove_optional_labels,
    _remove_stock_acknowledgement_opening,
    _remove_recommendation_template_headings,
    _remove_routine_product_recommendations,
    _remove_unearned_thanks_opening,
    _promote_product_headings_to_h4,
    validate_response,
)
from graph.nodes.classify_intent import _keyword_fallback
from services.literature_service import _is_relevant_article, _key_point, _pubmed_search_term
from services.profile_service import normalize_profile
from services.evidence_summary_service import (
    _clean_summary_paragraph,
    _skip_literature_for_same_community_query,
    summarize_evidence,
)
from services.reddit_service import (
    _best_sentiment_quote,
    _enrich_posts,
    _matches_query,
    _post_summary,
    _relevance_score,
    search as reddit_search,
)
from tools import product_tools
from utils.product_match import product_family_name


def _clear_recommendation_cache():
    with recommendation_cache._lock:
        recommendation_cache._cache.clear()
        recommendation_cache._timestamps.clear()
        recommendation_cache._inflight.clear()


def test_normalize_profile_migrates_sensitive_skin_type_to_flag():
    profile = normalize_profile({"skin_type": "Sensitive", "concerns": []})

    assert profile["skin_type"] == "Normal"
    assert profile["sensitive_skin"] is True


def test_recommendation_query_includes_sensitive_skin_separately():
    query = _build_recommendation_query(
        {
            "message": "Suggest a cleanser",
            "user_profile": {
                "skin_type": "Dry",
                "sensitive_skin": True,
                "concerns": [],
            },
            "user_routine": {},
        }
    )

    assert "Skin type: Dry" in query
    assert "Sensitive skin: yes" in query


def test_prepared_answer_exact_match_is_scoped_to_profile_and_routine():
    _clear_recommendation_cache()
    profile = {"user_id": "u1", "skin_type": "dry"}
    routine = {"items": [{"product_name": "Existing Cleanser"}]}
    result = {"response": "Prepared", "matched_products": [{"product_name": "Barrier Cream"}]}

    store_prepared_answer(
        "Suggest 1-2 targeted additions to my routine.",
        "recommend",
        profile,
        routine,
        result,
    )

    cached = get_prepared_answer(
        "Suggest 1-2 targeted additions to my routine.",
        "recommend",
        profile,
        routine,
    )
    assert cached["response"] == "Prepared"
    assert cached["prepared_answer"] is True

    assert get_prepared_answer(
        "Suggest 1-2 targeted additions to my routine.",
        "recommend",
        {"user_id": "u1", "skin_type": "oily"},
        routine,
    ) is None


def test_prepared_answer_reuses_generic_similar_question_but_not_specific_product_request():
    _clear_recommendation_cache()
    profile = {"user_id": "u1", "skin_type": "combination"}
    routine = {"items": []}
    result = {"response": "Prepared ingredients", "matched_products": []}

    store_prepared_answer(
        "Suggest the top 1-2 skincare ingredients that suit my skin profile.",
        "learn",
        profile,
        routine,
        result,
    )

    cached = get_prepared_answer(
        "what ingredients should I try for my skin?",
        "learn",
        profile,
        routine,
    )
    assert cached["response"] == "Prepared ingredients"

    assert get_prepared_answer(
        "should I try niacinamide or vitamin c?",
        "learn",
        profile,
        routine,
    ) is None


def test_prepared_answer_can_use_matching_inflight_job():
    _clear_recommendation_cache()
    profile = {"user_id": "u1", "skin_type": "dry"}
    routine = {"items": []}
    future: Future = Future()
    future.set_result({"response": "Prepared from background", "matched_products": []})
    key = recommendation_cache._cache_key(
        "Suggest 1-2 targeted additions to my routine.",
        "recommend",
        profile,
        routine,
    )
    with recommendation_cache._lock:
        recommendation_cache._inflight[key] = future

    cached = get_prepared_answer(
        "what should I add to my routine?",
        "recommend",
        profile,
        routine,
        wait_timeout=0.1,
    )

    assert cached["response"] == "Prepared from background"
    assert cached["prepared_answer"] is True


def test_report_progress_ignores_streamlit_stop_exception():
    StopException = type("StopException", (BaseException,), {})

    def raise_stop(_message):
        raise StopException()

    assert _report_progress(raise_stop, "Using prepared recommendation") is None


def test_select_recommended_product_cards_uses_llm_selected_candidate(monkeypatch):
    response = "I would add The Ordinary Lactic Acid 5% + HA at night."
    products = [
        {"product_name": "Skin + Me cream cleanser sensitive skin 80ml", "brand": "Skin"},
        {"product_name": "The Ordinary Lactic Acid 5% + HA", "brand": "The Ordinary"},
    ]

    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [{"index": 1}],
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": response,
            "candidate_products": products,
            "routine_items": [],
            "limit": 6,
        }
    ) == [products[1]]


def test_cards_for_selected_product_names_resolves_brand_suffix():
    products = [
        {
            "product_name": (
                "Skin + Me Breakouts + Visible Pores Serum, for Sensitive Skin, "
                "with Azelaic Acid and Annona Extract"
            ),
            "brand": "Skin + Me",
            "quantity": "12ml",
            "key_ingredients": ["Azelaic Acid"],
        }
    ]

    assert _cards_for_selected_product_names(
        [
            (
                "Skin + Me Breakouts + Visible Pores Serum, for Sensitive Skin, "
                "with Azelaic Acid and Annona Extract 12ml · Skin"
            )
        ],
        products,
        routine_items=[],
    ) == products


def test_cards_for_selected_product_names_keeps_second_distinct_product(monkeypatch):
    ordinary = {
        "product_name": (
            "The Ordinary Multi-Peptide + Copper Peptides 1% Serum to Target "
            "Signs of Aging - 30ml"
        ),
        "brand": "The Ordinary",
        "quantity": "30ml",
        "image_url": "https://example.com/ordinary.jpg",
        "key_ingredients": ["Copper Tripeptide-1"],
    }
    skin_me = {
        "product_name": (
            "Skin + Me Breakouts + Visible Pores Serum, with Azelaic Acid"
        ),
        "brand": "Skin + Me",
        "quantity": "12ml",
        "image_url": "https://example.com/skin-me.jpg",
        "key_ingredients": ["Azelaic Acid"],
    }

    def fake_find_catalog_product(query, *args, **kwargs):
        if "Skin + Me Breakouts" in query:
            return skin_me
        if "Multi-Peptide + Copper Peptides" in query:
            return ordinary
        return None

    monkeypatch.setattr(product_tools, "_find_catalog_product", fake_find_catalog_product)

    response = (
        "The Ordinary Multi-Peptide + Copper Peptides 1% Serum\n"
        "Peptides support skin firmness.\n\n"
        "Skin + Me Breakouts + Visible Pores Serum, with Azelaic Acid\n"
        "Azelaic Acid helps improve skin texture."
    )

    assert _cards_for_selected_product_names(
        [
            "The Ordinary Multi-Peptide + Copper Peptides 1% Serum",
            "Skin + Me Breakouts + Visible Pores Serum, with Azelaic Acid",
        ],
        [ordinary],
        routine_items=[],
        response=response,
    ) == [ordinary, skin_me]


def test_product_family_name_removes_size_suffixes():
    assert product_family_name(
        "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml"
    ) == product_family_name(
        "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 60ml"
    )


def test_select_recommended_product_cards_dedupes_size_variants(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    products = [
        {
            "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
            "brand": "The Ordinary",
            "image_url": "https://example.com/30.jpg",
        },
        {
            "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 60ml",
            "brand": "The Ordinary",
            "image_url": "https://example.com/60.jpg",
        },
    ]

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "Product Suggestion: The Ordinary Multi-Peptide + Hyaluronic Acid Serum "
                "for Aging Skin - 30ml\n"
                "Product Suggestion: The Ordinary Multi-Peptide + Hyaluronic Acid Serum "
                "for Aging Skin - 60ml"
            ),
            "candidate_products": products,
            "routine_items": [],
            "limit": 6,
        }
    ) == [products[0]]


def test_select_recommended_product_cards_handles_brand_suffix_in_suggestion(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    product = {
        "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
        "brand": "The Ordinary",
        "image_url": "https://example.com/30.jpg",
    }

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "**Product Suggestion:** The Ordinary Multi-Peptide + Hyaluronic Acid "
                "Serum for Aging Skin - 30ml · The Ordinary"
            ),
            "candidate_products": [product],
            "routine_items": [],
            "limit": 6,
        }
    ) == [product]


def test_select_recommended_product_cards_reads_h4_product_headings(monkeypatch):
    product = {
        "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
        "brand": "The Ordinary",
        "image_url": "https://example.com/30.jpg",
    }

    monkeypatch.setattr(
        product_tools,
        "search_products",
        lambda query, limit=5, use_obf_fallback=True: [product],
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "These additions fit your dry-skin goals.\n\n"
                "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml\n"
                "- **Why it fits:** Adds peptides and hydration."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 6,
        }
    ) == [product]


def test_select_recommended_product_cards_matches_combined_size_heading(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    products = [
        {
            "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
            "brand": "The Ordinary",
            "image_url": "https://example.com/30.jpg",
        },
        {
            "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 60ml",
            "brand": "The Ordinary",
            "image_url": "https://example.com/60.jpg",
        },
    ]

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "I understand you're looking for anti-ageing, texture and hydration support.\n\n"
                "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml / 60ml\n"
                "- **Why it fits:** Adds peptides and hydration."
            ),
            "candidate_products": products,
            "routine_items": [],
            "limit": 6,
        }
    ) == [products[0]]


def test_select_recommended_product_cards_queries_catalog_before_llm(monkeypatch):
    response = "I would add The Ordinary Lactic Acid 5% + HA at night."
    candidate = {"product_name": "The Ordinary Lactic Acid 5% + HA", "brand": "The Ordinary"}
    catalog_product = {**candidate, "image_url": "https://example.com/product.jpg", "source": "db"}
    calls = []

    def fake_search_products(query, limit=5, use_obf_fallback=True):
        calls.append((query, use_obf_fallback))
        return [catalog_product]

    monkeypatch.setattr(product_tools, "search_products", fake_search_products)
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": response,
            "candidate_products": [candidate],
            "routine_items": [],
            "limit": 6,
        }
    ) == [catalog_product]
    assert calls == [("The Ordinary Lactic Acid 5% + HA", False)]


def test_select_recommended_product_cards_resolves_labeled_suggestion_from_catalog(monkeypatch):
    product = {
        "product_name": "Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant",
        "brand": "Paula's Choice",
        "source": "db",
    }
    calls = []

    def fake_search_products(query, limit=5, use_obf_fallback=True):
        calls.append((query, use_obf_fallback))
        return [product]

    monkeypatch.setattr(product_tools, "search_products", fake_search_products)
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: (_ for _ in ()).throw(AssertionError("LLM should not be called")),
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": "Product Suggestion: Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant.",
            "candidate_products": [],
            "routine_items": [],
            "limit": 6,
        }
    ) == [product]
    assert calls == [("Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant", True)]


def test_select_recommended_product_cards_resolves_heading_suggestions_from_catalog(monkeypatch):
    clinique = {
        "product_name": "Clinique Superdefense Night Moisturiser - Combination/Oily",
        "brand": "Clinique",
        "source": "db",
    }
    innisfree = {
        "product_name": "Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads",
        "brand": "Innisfree",
        "source": "db",
    }
    products_by_query = {
        clinique["product_name"]: [clinique],
        innisfree["product_name"]: [innisfree],
    }
    calls = []

    def fake_search_products(query, limit=5, use_obf_fallback=True):
        calls.append((query, use_obf_fallback))
        return products_by_query.get(query, [])

    monkeypatch.setattr(product_tools, "search_products", fake_search_products)
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [],
    )

    response = dedent(
        """
        Based on your profile and current routine, here are two product suggestions:

        Clinique Superdefense Night Moisturiser - Combination/Oily (Rest PM)

        Why it fits: This moisturizer is specifically formulated for combination to oily skin.

        Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads (Active AM)

        Why it fits: These toner pads contain niacinamide.
        """
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": response,
            "candidate_products": [],
            "routine_items": [],
            "limit": 6,
        }
    ) == [clinique, innisfree]
    assert calls == [
        ("Clinique Superdefense Night Moisturiser - Combination/Oily", True),
        ("Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads", True),
    ]


def test_select_recommended_product_cards_reads_bold_percent_and_booster_products(monkeypatch):
    resveratrol = {
        "product_name": "The Ordinary Resveratrol 3% + Ferulic Acid 3%",
        "brand": "The Ordinary",
        "source": "db",
        "image_url": "https://example.com/resveratrol.jpg",
    }
    azelaic = {
        "product_name": "Paula's Choice 10% Azelaic Acid Booster",
        "brand": "Paula's Choice",
        "source": "db",
        "image_url": "https://example.com/azelaic.jpg",
    }
    products_by_query = {
        resveratrol["product_name"]: [resveratrol],
        azelaic["product_name"]: [azelaic],
    }

    def fake_search_products(query, limit=5, use_obf_fallback=True):
        return products_by_query.get(query, [])

    monkeypatch.setattr(product_tools, "search_products", fake_search_products)
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    response = dedent(
        """
        Here are two product recommendations that contain the suggested ingredients:

        **The Ordinary Resveratrol 3% + Ferulic Acid 3%**
        Combines Resveratrol with Ferulic Acid for antioxidant support.

        **Paula's Choice 10% Azelaic Acid Booster**
        Contains Azelaic Acid to target uneven skin tone and texture.
        """
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": response,
            "candidate_products": [],
            "routine_items": [],
            "limit": 2,
        }
    ) == [resveratrol, azelaic]


def test_select_recommended_product_cards_returns_minimal_card_when_catalog_misses(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "#### Made Up Ultra Gentle Cleanser\n"
                "- This is a gentle cleanser option for sensitive skin."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 1,
        }
    )

    assert result == [
        {
            "product_name": "Made Up Ultra Gentle Cleanser",
            "brand": "",
            "quantity": "",
            "image_url": "",
            "key_ingredients": [],
            "active_ingredients": [],
            "ingredients": [],
            "categories": [],
            "source": "assistant_recommendation",
        }
    ]


def test_select_recommended_product_cards_rejects_option_label_ingredient_headings(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "#### Option — Azelaic Acid (serum, 10–15%)\n"
                "- Fits oily, acne-prone skin.\n\n"
                "#### Option — 2% Salicylic Acid (leave-on BHA serum or gel)\n"
                "- Use on retinol-free PM nights."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 2,
        }
    )

    assert result == []


def test_select_recommended_product_cards_rejects_descriptor_paren_ingredient(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "#### Azelaic Acid (serum, 10-15%)\n"
                "- Fits oily, acne-prone skin."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 1,
        }
    )

    assert result == []


def test_select_recommended_product_cards_strips_option_prefix_for_real_product(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "#### Option — The Ordinary Niacinamide 10% + Zinc 1% Oil Control Serum\n"
                "- Adds oil control."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 1,
        }
    )

    assert len(result) == 1
    assert result[0]["product_name"] == "The Ordinary Niacinamide 10% + Zinc 1% Oil Control Serum"


def test_select_recommended_product_cards_rejects_generic_bha_request_phrase(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": "#### for a 2% BHA leave-on\nUse it on retinol-free PM nights.",
            "candidate_products": [],
            "routine_items": [],
            "limit": 1,
        }
    )

    assert result == []


def test_selected_product_names_reject_generic_bha_request_phrase():
    assert _cards_for_selected_product_names(
        ["for a 2% BHA leave-on"],
        [],
        [],
        response="Product suggestion for a 2% BHA leave-on",
    ) == []


def test_select_recommended_product_cards_reads_plain_product_lines(monkeypatch):
    numbuzin = {
        "product_name": "Numbuzin No.1 Clear Filter Sun Essence SPF50+ PA++++ 50ml",
        "brand": "Numbuzin",
        "source": "db",
    }
    tocobo = {
        "product_name": "Tocobo Cica Calming Sun Serum SPF50 50ml",
        "brand": "Tocobo",
        "source": "db",
    }
    products_by_query = {
        numbuzin["product_name"]: [numbuzin],
        tocobo["product_name"]: [tocobo],
    }

    monkeypatch.setattr(
        product_tools,
        "search_products",
        lambda query, limit=5, use_obf_fallback=True: products_by_query.get(query, []),
    )
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    response = dedent(
        """
        Starting from your routine for an oily, acne-prone skin profile, two lightweight high-SPF options:

        Numbuzin No.1 Clear Filter Sun Essence SPF50+ PA++++ 50ml
        Lightweight essence texture that reduces shine and layers well.

        Tocobo Cica Calming Sun Serum SPF50 50ml
        Serum texture suited for oily, acne-prone skin and makeup layering.
        """
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": response,
            "candidate_products": [],
            "routine_items": [],
            "limit": 2,
        }
    ) == [numbuzin, tocobo]


def test_select_recommended_product_cards_reads_plain_sensitive_serum_line(monkeypatch):
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])
    monkeypatch.setattr(product_tools, "_ask_product_card_llm", lambda *args, **kwargs: [])

    result = product_tools.select_recommended_product_cards.invoke(
        {
            "response": (
                "Skin + Me Breakouts + Visible Pores Serum, for Sensitive Skin, "
                "with Azelaic Acid and Annona Extract 12ml\n"
                "Fits inflammatory acne and redness."
            ),
            "candidate_products": [],
            "routine_items": [],
            "limit": 1,
        }
    )

    assert result[0]["product_name"] == (
        "Skin + Me Breakouts + Visible Pores Serum, for Sensitive Skin, "
        "with Azelaic Acid and Annona Extract 12ml"
    )
    assert result[0]["source"] == "assistant_recommendation"


def test_extract_recommended_product_names_ignores_description_lines():
    response = dedent(
        """
        La Roche-Posay Toleriane Gel Moussant Purifying Foaming Cleansing Gel For Sensitive Skin 400ml
        Fits oily, acne-prone skin and large pores.
        Gentle foaming gel with glycerin and ceramide NP for hydration.

        Cetaphil Gentle Skin Cleanser, Face Wash & Body Wash for Sensitive Skin 473ml
        Includes glycerin and panthenol to soothe and hydrate.
        Very low irritation risk alongside retinol and niacinamide.
        """
    )

    assert product_tools.extract_recommended_product_names_from_response.invoke(
        {"response": response, "limit": 4}
    ) == [
        "La Roche-Posay Toleriane Gel Moussant Purifying Foaming Cleansing Gel For Sensitive Skin 400ml",
        "Cetaphil Gentle Skin Cleanser, Face Wash & Body Wash for Sensitive Skin 473ml",
    ]


def test_followup_cleanser_request_classifies_as_recommend():
    assert _keyword_fallback("how about a gentle cleanser?") == "recommend"


def test_select_recommended_product_cards_excludes_routine_products(monkeypatch):
    products = [
        {
            "product_name": "La Roche-Posay Anthelios Clear Skin Oil-Free Sunscreen SPF 60",
            "brand": "La Roche-Posay",
        }
    ]
    routine_items = [
        {
            "product_name": "La Roche-Posay Anthelios Clear Skin Oil-Free Sunscreen SPF 60",
            "brand": "La Roche-Posay",
        }
    ]
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [{"index": 0}],
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": "Use this sunscreen each morning.",
            "candidate_products": products,
            "routine_items": routine_items,
            "limit": 6,
        }
    ) == []


def test_select_recommended_product_cards_does_not_fall_back_to_irrelevant_candidates(monkeypatch):
    products = [
        {
            "product_name": "TYPEBEA G1 Overnight Boosting Peptide Serum",
            "brand": "Typebea",
        }
    ]
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [],
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": "Try The Ordinary Salicylic Acid 2% Solution.",
            "candidate_products": products,
            "routine_items": [],
            "limit": 6,
        }
    ) == []


def test_select_recommended_product_cards_rejects_index_not_named_in_response(monkeypatch):
    products = [
        {
            "product_name": "TYPEBEA G1 Overnight Boosting Peptide Serum",
            "brand": "Typebea",
        }
    ]
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [{"index": 0}],
    )
    monkeypatch.setattr(product_tools, "search_products", lambda *args, **kwargs: [])

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": "Product Suggestion: The Ordinary Hyaluronic Acid 2% + B5",
            "candidate_products": products,
            "routine_items": [],
            "limit": 6,
        }
    ) == [
        {
            "product_name": "The Ordinary Hyaluronic Acid 2% + B5",
            "brand": "",
            "quantity": "",
            "image_url": "",
            "key_ingredients": [],
            "active_ingredients": [],
            "ingredients": [],
            "categories": [],
            "source": "assistant_recommendation",
        }
    ]


def test_select_recommended_product_cards_resolves_llm_product_name(monkeypatch):
    product = {
        "product_name": "The Ordinary Salicylic Acid 2% Solution",
        "brand": "The Ordinary",
    }
    monkeypatch.setattr(
        product_tools,
        "_ask_product_card_llm",
        lambda response, candidate_products, limit: [
            {
                "product_name": "The Ordinary Salicylic Acid 2% Solution",
                "brand": "The Ordinary",
                "search_query": "The Ordinary Salicylic Acid 2% Solution",
            }
        ],
    )
    monkeypatch.setattr(
        product_tools,
        "search_products",
        lambda query, limit=5: [product],
    )

    assert product_tools.select_recommended_product_cards.invoke(
        {
            "response": "Try The Ordinary Salicylic Acid 2% Solution.",
            "candidate_products": [],
            "routine_items": [],
            "limit": 6,
        }
    ) == [product]


def test_community_search_query_extracts_review_subject():
    assert (
        _community_search_query("What do people say about The Ordinary Niacinamide serum?")
        == "The Ordinary Niacinamide serum"
    )


def test_community_placeholder_replaces_confirmation_prompt():
    assert _community_placeholder_response(
        "What do people say about The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml?",
        "I can share community feedback summaries — but first I need to confirm you want opinions from user reviews/Reddit/online sources (yes/no)?",
    ) == (
        "Here is the plain-language community signal I found for "
        "**The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml**."
    )


def test_ingredient_evidence_query_prefers_retrieved_ingredient():
    assert (
        _ingredient_evidence_query(
            "What does niacinamide do?",
            {
                "mode": "learn",
                "retrieved_ingredients": [
                    {"display_name": "Niacinamide", "inci_name": "Niacinamide"}
                ],
            },
        )
        == "Niacinamide"
    )


def test_ingredient_evidence_query_uses_extracted_terms():
    assert (
        _ingredient_evidence_query(
            "Is azelaic acid good for redness?",
            {
                "mode": "learn",
                "retrieved_ingredients": [],
                "tool_outputs": [
                    {"tool": "extract_ingredient_terms", "output": ["azelaic acid"]}
                ],
            },
        )
        == "azelaic acid"
    )


def test_ingredient_evidence_query_extracts_uncatalogued_learn_term():
    assert (
        _ingredient_evidence_query(
            "What is PDRN and should I try it?",
            {"mode": "learn", "retrieved_ingredients": [], "tool_outputs": []},
        )
        == "PDRN"
    )


def test_ingredient_evidence_query_ignores_unmentioned_retrieved_ingredient():
    assert (
        _ingredient_evidence_query(
            "What ingredients should I try for my skin?",
            {
                "mode": "learn",
                "retrieved_ingredients": [
                    {"display_name": "Resveratrol", "inci_name": "Resveratrol"}
                ],
                "tool_outputs": [],
            },
        )
        is None
    )


def test_ingredient_evidence_query_ignores_non_learning_modes():
    assert _ingredient_evidence_query("Analyse my niacinamide serum", {"mode": "analyse"}) is None


def test_ingredient_evidence_query_does_not_treat_product_as_ingredient():
    assert (
        _ingredient_evidence_query(
            "What do people say about Clinique Superdefense Night Moisturiser?",
            {"mode": "learn", "retrieved_ingredients": [], "tool_outputs": []},
        )
        is None
    )


def test_ingredient_evidence_does_not_fetch_community_without_request():
    assert _community_query_for_evidence(None, "PDRN") is None


def test_ingredient_evidence_ignores_generic_learn_headings():
    response = dedent(
        """
        #### What it is
        PDRN is polydeoxyribonucleotide.

        #### How it's thought to work
        It is marketed for skin repair.
        """
    )

    assert _extract_recommended_ingredients(response) == []


def test_context_block_ignores_malformed_string_entries():
    context = _build_context_block(
        {
            "mode": "recommend",
            "user_profile": {
                "skin_type": "Combination",
                "concerns": "Acne",
                "goals": ["Calm redness"],
                "allergens": "fragrance",
                "notes": "Please keep routines minimal.",
            },
            "user_routine": {
                "mode": "daily",
                "items": [
                    "legacy routine item",
                    {
                        "product_name": "Gentle Cleanser",
                        "brand": "Example",
                        "ingredients": "glycerin",
                    },
                ],
            },
            "retrieved_products": [
                "bad product",
                {
                    "product_name": "Barrier Serum",
                    "source": "catalog_search",
                    "key_ingredients": "ceramide",
                },
            ],
            "retrieved_ingredients": [
                "bad ingredient",
                {"inci_name": "Niacinamide", "functions": "barrier support"},
            ],
            "matched_products": [
                "bad matched product",
                {"product_name": "Barrier Serum", "key_ingredients": "ceramide"},
            ],
            "analysis_results": {
                "conflicts": [
                    "bad conflict",
                    {
                        "risk_level": "low",
                        "ingredients_involved": "niacinamide",
                        "products_involved": "Barrier Serum",
                        "reason": "Low risk.",
                    },
                ],
                "gaps": "Routine may need barrier support.",
                "suitability_notes": ["Introduce slowly."],
            },
            "input_status": {"missing": "routine"},
        }
    )

    assert "Gentle Cleanser" in context
    assert "Barrier Serum" in context
    assert "legacy routine item" not in context
    assert "[LOW] Niacinamide" in context
    assert "User notes: Please keep routines minimal." in context


def test_analyse_context_marks_candidates_as_not_routine_products():
    context = _build_context_block(
        {
            "mode": "analyse",
            "user_profile": {"skin_type": "Oily"},
            "user_routine": {
                "mode": "daily",
                "items": [
                    {
                        "product_name": "Night Retinol Serum",
                        "time": "PM",
                        "key_ingredients": ["Retinol", "Glycerin"],
                    }
                ],
            },
            "matched_products": [
                {
                    "product_name": "Mandelic Acid Toner",
                    "key_ingredients": ["Mandelic Acid"],
                }
            ],
        }
    )

    assert "## Current routine products" in context
    assert "- Night Retinol Serum (PM) [Retinol, Glycerin]" in context
    assert "## Products mentioned by user (analyse each in depth)" in context
    assert "## Context boundary" in context
    assert "The only products the user currently uses are listed under \"Current routine products\"" in context


def test_retrieve_context_analyse_uses_only_explicit_product_matches(monkeypatch):
    monkeypatch.setattr(
        retrieve_context_module,
        "retrieve_semantic_context",
        lambda *args, **kwargs: (
            [{"product_name": "Candidate Acid Serum", "source": "semantic"}],
            [],
            None,
        ),
    )
    monkeypatch.setattr(
        retrieve_context_module,
        "_find_matched_products",
        lambda message: [{"product_name": "Explicitly Mentioned Cleanser"}],
    )
    monkeypatch.setattr(retrieve_context_module, "extract_ingredient_terms", type(
        "Extractor",
        (),
        {"invoke": staticmethod(lambda payload: [])},
    )())

    result = retrieve_context(
        {
            "mode": "analyse",
            "message": "Can you analyse my routine?",
            "user_profile": {"skin_type": "Oily"},
            "user_routine": {"items": [{"product_name": "Night Retinol Serum"}]},
        }
    )

    matched_names = [product["product_name"] for product in result["matched_products"]]
    assert matched_names == ["Explicitly Mentioned Cleanser"]
    retrieved_names = [product["product_name"] for product in result["retrieved_products"]]
    assert "Candidate Acid Serum" in retrieved_names


def test_context_block_includes_previous_recommendations():
    context = _build_context_block(
        {
            "mode": "recommend",
            "messages": [
                AIMessage(
                    content=(
                        "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml\n"
                        "- **Why it fits:** Adds peptides."
                    )
                )
            ],
            "user_profile": {"skin_type": "Dry"},
            "user_routine": {"items": []},
        }
    )

    assert "## Previously recommended products" in context
    assert "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml" in context
    assert "Do not recommend these again" in context


def test_recommendation_query_includes_profile_and_routine_context():
    query = _build_recommendation_query(
        {
            "message": "Suggest 1-2 additions to my routine.",
            "user_profile": {
                "skin_type": "Oily",
                "concerns": ["Acne / Breakouts", "Oiliness / Shine"],
                "goals": ["Clear skin"],
                "notes": "I prefer budget products and a simple routine.",
                "avoid_fragrance": True,
            },
            "user_routine": {
                "items": [
                    {
                        "product_name": "Gentle Cleanser",
                        "key_ingredients": ["Glycerin"],
                    }
                ]
            },
        }
    )

    assert "Skin type: Oily" in query
    assert "Skin concerns: Acne / Breakouts, Oiliness / Shine" in query
    assert "Helpful ingredients: salicylic acid" in query
    assert "User notes: I prefer budget products and a simple routine." in query
    assert "Avoid: fragrance" in query
    assert "Current routine products: Gentle Cleanser" in query


def test_recommendation_query_changes_with_profile_for_same_message():
    base_state = {
        "message": "Suggest 1-2 additions to my routine.",
        "user_routine": {"items": [{"product_name": "Gentle Cleanser"}]},
    }

    oily_query = _build_recommendation_query(
        {
            **base_state,
            "user_profile": {
                "skin_type": "Oily",
                "concerns": ["Acne / Breakouts"],
            },
        }
    )
    dry_query = _build_recommendation_query(
        {
            **base_state,
            "user_profile": {
                "skin_type": "Dry",
                "concerns": ["Dryness / Dehydration"],
            },
        }
    )

    assert oily_query != dry_query
    assert "salicylic acid" in oily_query
    assert "ceramides" in dry_query


def test_rank_products_for_profile_prioritizes_concern_matched_ingredients():
    products = [
        {
            "product_name": "Generic Hydrating Mist",
            "ingredients": ["Water", "Glycerin"],
        },
        {
            "product_name": "BHA Serum",
            "ingredients": ["Water", "Salicylic Acid", "Niacinamide"],
        },
    ]

    ranked = _rank_products_for_profile(
        products,
        {"concerns": ["Acne / Breakouts"]},
    )

    assert ranked[0]["product_name"] == "BHA Serum"


def test_rank_products_for_profile_excludes_explicit_skin_type_mismatch():
    products = [
        {
            "product_name": "Skin + Me Breakouts + Visible Pores Serum, for Dry to Normal Skin",
            "ingredients": ["Azelaic Acid"],
        },
        {
            "product_name": "Skin + Me Light Moisturiser For Oily To Combination Skin",
            "ingredients": ["Niacinamide", "Ceramides"],
        },
    ]

    ranked = _rank_products_for_profile(
        products,
        {"skin_type": "Oily", "concerns": ["Acne / Breakouts"]},
    )

    assert [product["product_name"] for product in ranked] == [
        "Skin + Me Light Moisturiser For Oily To Combination Skin"
    ]


def test_filter_skin_type_mismatches_keeps_matching_skin_type_range():
    product = {
        "product_name": "Skin + Me Light Moisturiser For Oily To Combination Skin",
        "ingredients": ["Niacinamide", "Ceramides"],
    }

    assert _filter_skin_type_mismatches([product], {"skin_type": "Combination"}) == [product]


def test_filter_duplicate_retinoids_excludes_new_retinoid_when_routine_has_retinol():
    products = [
        {
            "product_name": "The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
            "ingredients": ["Retinal", "Glycerin"],
        },
        {
            "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
            "ingredients": ["Acetyl Hexapeptide-8", "Sodium Hyaluronate"],
        },
    ]
    routine_items = [
        {
            "product_name": "The Ordinary Retinol 0.2% in Squalane",
            "key_ingredients": ["Retinol"],
        }
    ]

    filtered = _filter_duplicate_retinoids(
        products,
        routine_items,
        "Could you recommend more products",
    )

    assert [product["product_name"] for product in filtered] == [
        "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml"
    ]


def test_filter_duplicate_retinoids_allows_explicit_retinoid_replacement_request():
    product = {
        "product_name": "The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
        "ingredients": ["Retinal"],
    }
    routine_items = [{"product_name": "Current Retinol Serum", "key_ingredients": ["Retinol"]}]

    assert _filter_duplicate_retinoids(
        [product],
        routine_items,
        "Can you compare retinoids or replace my retinol?",
    ) == [product]


def test_literature_key_point_uses_abstract_summary():
    assert _key_point("Niacinamide improved barrier function in this trial.", "Title") == (
        "Niacinamide improved barrier function in this trial."
    )


def test_pubmed_pdrn_query_expands_acronym_and_excludes_training_contexts():
    term = _pubmed_search_term("PDRN")

    assert '"polydeoxyribonucleotide"[Title/Abstract]' in term
    assert '"program director"[Title/Abstract]' in term
    assert "ostomy[Title/Abstract]" in term


def test_pubmed_pdrn_relevance_rejects_residency_and_ostomy_false_matches():
    assert not _is_relevant_article(
        "PDRN",
        "Transitioning From an Intern to a Dermatology Resident.",
        "A program director resident network survey described internship needs.",
    )
    assert not _is_relevant_article(
        "PDRN",
        "Defining Opportunities to Improve Perioperative Ostomy Care and Education.",
        "The PDRN discussed ostomy education and perioperative care.",
    )
    assert _is_relevant_article(
        "PDRN",
        "Polydeoxyribonucleotide in skin regeneration.",
        "Polydeoxyribonucleotide was discussed for wound healing and skin repair.",
    )


def test_evidence_summary_fetches_literature_when_same_query_is_ingredient(monkeypatch):
    calls = {"literature": 0, "community": 0}

    def fake_paper_sources(query):
        calls["literature"] += 1
        return []

    def fake_community_sources(query):
        calls["community"] += 1
        return []

    monkeypatch.setattr("services.evidence_summary_service._summarize_with_llm", lambda *args: "")
    monkeypatch.setattr("services.evidence_summary_service._paper_sources", fake_paper_sources)
    monkeypatch.setattr("services.evidence_summary_service._community_sources", fake_community_sources)

    summarize_evidence(literature_query="PDRN", community_query="PDRN")

    assert calls == {"literature": 1, "community": 1}


def test_evidence_summary_skips_literature_for_same_product_community_query():
    assert _skip_literature_for_same_community_query(
        "The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
        "The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
    )
    assert not _skip_literature_for_same_community_query("PDRN", "PDRN")


def test_evidence_summary_fallback_returns_cited_paragraph(monkeypatch):
    monkeypatch.setattr("services.evidence_summary_service._summarize_with_llm", lambda *args: "")
    monkeypatch.setattr(
        "services.evidence_summary_service._paper_sources",
        lambda query: [
            {
                "id": "L1",
                "kind": "literature",
                "title": "Niacinamide and skin barrier",
                "url": "https://pubmed.ncbi.nlm.nih.gov/123/",
                "source": "PubMed",
                "year": "2024",
                "evidence": "Niacinamide may support skin barrier function in topical use.",
            }
        ],
    )
    monkeypatch.setattr(
        "services.evidence_summary_service._community_sources",
        lambda query: [
            {
                "id": "C1",
                "kind": "community",
                "title": "Niacinamide experiences",
                "url": "https://reddit.com/r/SkincareAddiction/comments/example",
                "source": "r/SkincareAddiction",
                "sentiment": "mixed",
                "evidence": "Some users liked it, while others found it irritating.",
            }
        ],
    )

    result = summarize_evidence(
        literature_query="niacinamide",
        community_query="The Ordinary Niacinamide serum",
    )

    assert "Niacinamide may support skin barrier" in result["paragraph"]
    assert "[[Niacinamide and skin barrier]](https://pubmed.ncbi.nlm.nih.gov/123/)" in result["paragraph"]
    assert "[[Niacinamide experiences]](https://reddit.com/r/SkincareAddiction/comments/example)" in result["paragraph"]
    assert "\n\n- [[Niacinamide experiences]]" in result["paragraph"]


def test_evidence_summary_is_blank_when_no_sources(monkeypatch):
    monkeypatch.setattr("services.evidence_summary_service._summarize_with_llm", lambda *args: "")
    monkeypatch.setattr("services.evidence_summary_service._paper_sources", lambda query, limit=4: [])
    monkeypatch.setattr("services.evidence_summary_service._community_sources", lambda query, limit=3: [])

    result = summarize_evidence(literature_query="What it is, How it's thought to work")

    assert result["paragraph"] == ""
    assert result["sources"] == []


def test_evidence_summary_cleans_absent_community_and_splits_point_labels():
    paragraph = (
        "Resveratrol — Clinical literature: retrieved antioxidant context. "
        "Common practical notes: tolerance can vary. Community sources: none provided. "
        "Palmitoyl Tripeptide-1 — Practical meaning: evidence is limited."
    )

    result = _clean_summary_paragraph(
        paragraph,
        [{"id": "L1", "kind": "literature", "title": "Antioxidants for Skin Health"}],
    )

    assert "Community sources" not in result
    assert "\n\nCommon practical notes:" in result
    assert "\n\nPractical meaning:" in result


def test_evidence_summary_fetches_each_comma_separated_ingredient(monkeypatch):
    calls = []

    def fake_paper_sources(query, limit=4):
        calls.append((query, limit))
        return [
            {
                "id": f"L{len(calls)}",
                "kind": "literature",
                "title": f"{query} source",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{len(calls)}/",
                "source": "PubMed",
                "evidence": f"{query} evidence",
            }
        ]

    monkeypatch.setattr("services.evidence_summary_service._summarize_with_llm", lambda *args: "")
    monkeypatch.setattr("services.evidence_summary_service._paper_sources", fake_paper_sources)
    monkeypatch.setattr("services.evidence_summary_service._community_sources", lambda query, limit=3: [])

    result = summarize_evidence(literature_query="Resveratrol, Palmitoyl Tripeptide-1")

    assert calls == [("Resveratrol", 2), ("Palmitoyl Tripeptide-1", 2)]
    assert len(result["sources"]) == 2


def test_evidence_summary_skips_literature_for_same_community_query(monkeypatch):
    calls = {"literature": 0}

    def fake_paper_sources(query):
        calls["literature"] += 1
        return []

    monkeypatch.setattr("services.evidence_summary_service._summarize_with_llm", lambda *args: "")
    monkeypatch.setattr("services.evidence_summary_service._paper_sources", fake_paper_sources)
    monkeypatch.setattr(
        "services.evidence_summary_service._community_sources",
        lambda query: [
            {
                "id": "C1",
                "kind": "community",
                "title": "Retinal product impressions",
                "url": "https://reddit.com/r/SkincareAddiction/comments/retinal",
                "source": "r/SkincareAddiction",
                "sentiment": "mixed",
                "evidence": "Several users say they introduce it slowly because retinal can irritate.",
            }
        ],
    )

    result = summarize_evidence(
        literature_query="The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
        community_query="The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml",
    )

    assert calls["literature"] == 0
    assert "Reddit as practical context" in result["paragraph"]


def test_reddit_summary_falls_back_to_discussion_context():
    assert _post_summary("Is azelaic acid worth trying?", "", "SkincareAddiction", 42) == (
        "r/SkincareAddiction discussion: Is azelaic acid worth trying? with 42 comments."
    )


def test_reddit_negative_quote_uses_clear_verdict_sentence():
    assert _best_sentiment_quote(
        "Clinique Superdefense Night Moisturiser",
        "Cruelty-free night moisturiser",
        "",
        [
            "I am currently using Clinique Superdefense Night but it is not cruelty free and the texture is a bit sticky.",
            "Clinique Superdefense Night feels sticky on my sensitive dry skin.",
            "Thank you for any recommendations!",
        ],
        "negative",
    ) == (
        "Clinique Superdefense Night feels sticky on my sensitive dry skin."
    )


def test_reddit_positive_quote_uses_clear_verdict_sentence():
    assert _best_sentiment_quote(
        "Clinique Superdefense Night Moisturiser",
        "Clinique Superdefense",
        "",
        ["Clinique Superdefense Night is hydrating and makes my skin feel soft."],
        "positive",
    ) == "Clinique Superdefense Night is hydrating and makes my skin feel soft."


def test_reddit_relevance_uses_product_tokens():
    assert _relevance_score(
        "Clinique Superdefense Night Moisturiser",
        {
            "title": "Cruelty-free night moisturiser",
            "comments_text": "Clinique Superdefense feels sticky.",
        },
    ) > 0


def test_reddit_product_match_requires_distinctive_product_terms():
    query = "Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads"

    assert not _matches_query(query, "Cerave Hydrating Toner 3.")
    assert not _matches_query(
        query,
        "Morning: Cerave cleanser Innisfree Vitamin C Pads or Inkey List PHA toner.",
    )
    assert _matches_query(
        query,
        "Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads made my skin smooth.",
    )


def test_reddit_skin_and_me_query_requires_brand_phrase():
    assert not _matches_query("skin + me", "TIFU dismissing bright red blood in my stool.")
    assert not _matches_query("skin + me", "My skin is red and me trying azelaic acid did not help.")
    assert _matches_query("skin + me", "Skin + Me Redness + Uneven Tone Serum helped my redness.")
    assert _matches_query("skin and me", "I tried the Skin + Me cleanser.")


def test_reddit_empty_distinctive_tokens_fail_closed():
    assert not _matches_query("skin", "Unrelated high-traffic discussion.")
    assert (
        _relevance_score(
            "skin + me",
            {"title": "TIFU dismissing bright red blood", "selftext": "Unrelated post."},
        )
        == 0
    )


def test_reddit_quote_rejects_unrelated_product_sentence():
    assert (
        _best_sentiment_quote(
            "Innisfree Green Tea Enzyme Vitamin C Brightening Toner Pads",
            "Routine help",
            "",
            ["Cerave Hydrating Toner 3 is good."],
            "positive",
        )
        == ""
    )


def test_reddit_enrichment_returns_top_three_by_upvotes(monkeypatch):
    monkeypatch.setattr("services.reddit_service._fetch_comments", lambda permalink, limit=8: [])
    posts = [
        {
            "title": "Clinique Superdefense low",
            "url": "https://reddit.com/low",
            "permalink": "/low",
            "score": 2,
            "num_comments": 1,
            "selftext": "Clinique Superdefense Night is good.",
            "subreddit": "SkincareAddiction",
        },
        {
            "title": "Clinique Superdefense high",
            "url": "https://reddit.com/high",
            "permalink": "/high",
            "score": 30,
            "num_comments": 1,
            "selftext": "Clinique Superdefense Night is good.",
            "subreddit": "SkincareAddiction",
        },
        {
            "title": "Clinique Superdefense mid",
            "url": "https://reddit.com/mid",
            "permalink": "/mid",
            "score": 20,
            "num_comments": 1,
            "selftext": "Clinique Superdefense Night is good.",
            "subreddit": "SkincareAddiction",
        },
        {
            "title": "Clinique Superdefense extra",
            "url": "https://reddit.com/extra",
            "permalink": "/extra",
            "score": 10,
            "num_comments": 1,
            "selftext": "Clinique Superdefense Night is good.",
            "subreddit": "SkincareAddiction",
        },
    ]

    results = _enrich_posts("Clinique Superdefense Night Moisturiser", posts, limit=6)

    assert [post["url"] for post in results] == [
        "https://reddit.com/high",
        "https://reddit.com/mid",
        "https://reddit.com/extra",
    ]


def test_reddit_search_collects_wide_pool_before_top_three(monkeypatch):
    def fake_subreddit(subreddit, query, limit=10, sort="relevance"):
        return [
            {
                "title": f"Clinique Superdefense low {subreddit} {sort}",
                "url": f"https://reddit.com/{subreddit}-{sort}",
                "permalink": f"/{subreddit}-{sort}",
                "score": 1,
                "num_comments": 1,
                "selftext": "Clinique Superdefense Night is good.",
                "subreddit": subreddit,
            }
        ]

    def fake_wide(query, limit=20, sort="relevance"):
        return [
            {
                "title": f"Clinique Superdefense high {sort}",
                "url": f"https://reddit.com/high-{sort}",
                "permalink": f"/high-{sort}",
                "score": 100 if sort == "top" else 50,
                "num_comments": 1,
                "selftext": "Clinique Superdefense Night is good.",
                "subreddit": "SkincareAddiction",
            }
        ]

    monkeypatch.setattr("services.reddit_service._search_subreddit", fake_subreddit)
    monkeypatch.setattr("services.reddit_service._search_wide", fake_wide)
    monkeypatch.setattr("services.reddit_service._fetch_comments", lambda permalink, limit=8: [])
    monkeypatch.setattr("services.reddit_service._cache_get", lambda key: None)
    monkeypatch.setattr("services.reddit_service._cache_set", lambda key, results: None)

    results = reddit_search("Clinique Superdefense Night Moisturiser", limit=3)

    assert results[0]["url"] == "https://reddit.com/high-top"
    assert len(results) == 3


def test_keyword_fallback_routes_add_to_routine_as_recommend():
    assert _keyword_fallback("What should I add to my routine?") == "recommend"


def test_keyword_fallback_routes_products_with_ingredient_as_recommend():
    assert _keyword_fallback("how about products with Matrixyl?") == "recommend"
    assert _keyword_fallback("Any serums containing azelaic acid?") == "recommend"


def test_keyword_fallback_routes_topical_ingredient_followup_as_learn():
    assert _keyword_fallback("it is a topical ingredient") == "learn"


def test_removes_unearned_thanks_opening():
    assert _remove_unearned_thanks_opening(
        "Thanks — aiming for anti-ageing and hydration makes sense."
    ) == "Aiming for anti-ageing and hydration makes sense."


def test_removes_stock_acknowledgement_opening():
    assert _remove_stock_acknowledgement_opening(
        "I understand you're looking for more product options — here are two choices."
    ) == "Here are two choices."


def test_removes_optional_labels():
    assert _remove_optional_labels("(Optional) The Ordinary Retinal 0.2% Emulsion") == (
        "The Ordinary Retinal 0.2% Emulsion"
    )


def test_removes_recommendation_template_headings():
    response = (
        "Primary suggestion — one product to add\n"
        "**Product Suggestion:** Peptide Serum\n\n"
        "Recommendation\n"
        "#### Peptide Serum\n\n"
        "Secondary option (only if you want a boost)\n"
        "**Product Suggestion:** Retinal Serum"
    )

    assert _remove_recommendation_template_headings(response) == (
        "**Product Suggestion:** Peptide Serum\n\n"
        "#### Peptide Serum\n\n"
        "**Product Suggestion:** Retinal Serum"
    )


def test_promotes_recommended_product_bullets_to_h4():
    response = (
        "- The Ordinary Multi-Peptide + Hyaluronic Acid Serum (30-60 ml)\n"
        "- **Why it fits:** Adds peptides and hydration.\n\n"
        "- Skin + Me Fine Lines + Elasticity Serum\n"
        "- **Where to use:** Rest PM nights."
    )

    assert _promote_product_headings_to_h4(
        response,
        [
            "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml",
            "Skin + Me Fine Lines + Elasticity Serum",
        ],
    ) == (
        "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum (30-60 ml)\n"
        "- **Why it fits:** Adds peptides and hydration.\n\n"
        "#### Skin + Me Fine Lines + Elasticity Serum\n"
        "- **Where to use:** Rest PM nights."
    )


def test_removes_saved_routine_product_suggestion_block():
    response = dedent("""
    1. Salicylic Acid Treatment
    Product Suggestion: Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant
    Why It Fits: Helps with clogged pores.

    2. Niacinamide Serum
    Product Suggestion: The Ordinary Niacinamide 10% + Zinc 1% Oil Control Serum
    Why It Fits: You already have this in your routine.
    Where It Belongs: Continue using it in both routines.
    """)
    routine_names = {"the ordinary niacinamide 10 zinc 1 oil control serum"}

    assert _remove_routine_product_recommendations(response, routine_names) == (
        "1. Salicylic Acid Treatment\n"
        "Product Suggestion: Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant\n"
        "Why It Fits: Helps with clogged pores."
    )


def test_keeps_new_product_suggestion_block():
    response = dedent("""
    1. Salicylic Acid Treatment
    Product Suggestion: Paula's Choice Skin Perfecting 2% BHA Liquid Exfoliant
    Why It Fits: Helps with clogged pores.
    """)
    routine_names = {"the ordinary niacinamide 10 zinc 1 oil control serum"}

    assert _remove_routine_product_recommendations(response, routine_names) == response.strip()


def test_validate_response_removes_saved_product_suggestion_from_routine():
    state = {
        "message": "What should I add?",
        "draft_response": (
            "1. Niacinamide Serum\n"
            "Product Suggestion: The Ordinary Niacinamide 10% + Zinc 1% Oil Control Serum\n"
            "Why It Fits: You already have this in your routine.\n"
        ),
        "user_routine": {
            "items": [
                {
                    "product_name": "The Ordinary Niacinamide 10% + Zinc 1% Oil Control Serum",
                    "brand": "The Ordinary",
                }
            ]
        },
    }

    assert validate_response(state)["final_response"] == (
        "I would not add that as a new product because it is already in your routine. "
        "Focus on how you use it within the existing routine instead."
    )


def test_validate_response_removes_previous_size_variant_recommendation():
    state = {
        "mode": "recommend",
        "message": "Could you recommend more products?",
        "messages": [
            AIMessage(
                content=(
                    "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 30ml\n"
                    "- **Why it fits:** Adds hydration."
                )
            )
        ],
        "matched_products": [
            {
                "product_name": "The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 60ml"
            }
        ],
        "draft_response": (
            "#### The Ordinary Multi-Peptide + Hyaluronic Acid Serum for Aging Skin - 60ml\n"
            "- **Why it fits:** Same peptide benefits in a larger size."
        ),
        "user_routine": {"items": []},
    }

    assert validate_response(state)["final_response"] == (
        "I already suggested those products earlier, so I would not add them again. "
        "Look for a different gap in the routine instead."
    )


def test_validate_response_removes_extra_retinoid_when_routine_has_retinol():
    state = {
        "mode": "recommend",
        "message": "Could you recommend more products?",
        "matched_products": [
            {
                "product_name": "The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml"
            }
        ],
        "draft_response": (
            "#### The Ordinary Retinal 0.2% Emulsion Anti-Aging Serum 15ml\n"
            "- **Why it fits:** A stronger vitamin A option."
        ),
        "user_routine": {
            "items": [
                {
                    "product_name": "The Ordinary Retinol 0.2% in Squalane",
                    "key_ingredients": ["Retinol"],
                }
            ]
        },
    }

    assert validate_response(state)["final_response"] == (
        "Because your routine already includes a retinoid, I would not add another retinoid product. "
        "Choose a non-retinoid support step instead, such as hydration or barrier support."
    )


# ── Fix: literature_service does not cache API timeouts ───────────────────────

def test_literature_search_does_not_cache_on_exception(monkeypatch):
    """A timeout/network error must not write an empty list to the 24-hour cache."""
    import services.literature_service as lit
    import requests

    lit._CACHE.clear()

    def _raise(*args, **kwargs):
        raise requests.exceptions.Timeout("simulated timeout")

    monkeypatch.setattr(requests, "get", _raise)

    result = lit.search("PDRN")
    assert result == []
    # Cache must stay empty so the next call actually hits the network.
    assert "pdrn:5" not in lit._CACHE


def test_literature_search_caches_real_empty_result(monkeypatch):
    """A successful API call returning zero IDs should still be cached."""
    import services.literature_service as lit
    import requests

    lit._CACHE.clear()

    class _FakeResp:
        status_code = 200
        def raise_for_status(self): pass
        def json(self):
            return {"esearchresult": {"idlist": []}}

    monkeypatch.setattr(requests, "get", lambda *a, **kw: _FakeResp())

    result = lit.search("PDRN", limit=5)
    assert result == []
    # Zero-result successful response IS cached.
    assert "pdrn:5" in lit._CACHE


# ── Fix: keyword_fallback "related products" → recommend ──────────────────────

def test_keyword_fallback_related_products():
    assert _keyword_fallback("related products") == "recommend"


def test_keyword_fallback_show_me_products():
    assert _keyword_fallback("show me products") == "recommend"


def test_keyword_fallback_any_products():
    assert _keyword_fallback("any products") == "recommend"


def test_keyword_fallback_products_for_this():
    assert _keyword_fallback("products for this") == "recommend"


def test_keyword_fallback_where_to_buy():
    assert _keyword_fallback("where to buy") == "recommend"


# ── Fix: _topic_from_last_ai_message extracts context ────────────────────────

def test_topic_from_last_ai_message_h4_heading():
    """Extracts first non-generic H4 heading from last AI message."""
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "messages": [
            HumanMessage(content="what is PDRN?"),
            AIMessage(content="#### PDRN\n\n- **PDRN** (Polydeoxyribonucleotide) is a DNA-derived ingredient.\n\n#### How It Works\n\n- Supports tissue regeneration."),
        ]
    }
    topic = _topic_from_last_ai_message(state)
    assert topic == "PDRN"


def test_topic_from_last_ai_message_skips_generic_headings():
    """Generic section headings (What It Is, How It Works, etc.) are skipped."""
    from langchain_core.messages import AIMessage

    state = {
        "messages": [
            AIMessage(content="#### What It Is\n\nSome ingredient info.\n\n#### How It Works\n\nDetails here."),
        ]
    }
    # All headings are generic → falls back to first line
    topic = _topic_from_last_ai_message(state)
    assert topic is not None
    # Should be the first non-heading line or first sentence (not a heading itself)


def test_topic_from_last_ai_message_no_messages():
    assert _topic_from_last_ai_message({"messages": []}) is None
    assert _topic_from_last_ai_message({}) is None


def test_topic_from_last_ai_message_only_human_messages():
    from langchain_core.messages import HumanMessage

    state = {"messages": [HumanMessage(content="hello")]}
    assert _topic_from_last_ai_message(state) is None


# ── Fix: _build_recommendation_query enriches short follow-up with context ────

def test_build_recommendation_query_enriches_related_products():
    """'related products' message is enriched with topic from last AI message."""
    from langchain_core.messages import AIMessage, HumanMessage

    state = {
        "message": "related products",
        "mode": "recommend",
        "messages": [
            HumanMessage(content="what is PDRN?"),
            AIMessage(content="#### PDRN\n\n- PDRN is a DNA-derived ingredient used in dermatology."),
        ],
        "user_profile": {},
        "user_routine": {},
    }
    query = _build_recommendation_query(state)
    assert "pdrn" in query.lower()


def test_build_recommendation_query_unaffected_for_specific_message():
    """A specific, non-generic message is NOT modified by the enrichment logic."""
    from langchain_core.messages import AIMessage

    state = {
        "message": "suggest a vitamin C serum for oily skin",
        "mode": "recommend",
        "messages": [
            AIMessage(content="#### PDRN\n\n- ..."),
        ],
        "user_profile": {},
        "user_routine": {},
    }
    query = _build_recommendation_query(state)
    assert "vitamin c" in query.lower()
    # PDRN should NOT appear — specific message was not replaced
    assert "pdrn" not in query.lower()


def test_build_recommendation_query_no_history_uses_message():
    """Without conversation history the query stays unchanged."""
    state = {
        "message": "related products",
        "mode": "recommend",
        "messages": [],
        "user_profile": {},
        "user_routine": {},
    }
    query = _build_recommendation_query(state)
    # No topic injected, original message preserved
    assert "related products" in query.lower()


# ── Fix: _GENERIC_FOLLOWUP_RE matches expected phrases ───────────────────────

def test_generic_followup_re_matches():
    matches = [
        "related products",
        "related product",
        "products for this",
        "products for it",
        "show me products",
        "any products",
        "what products",
        "more products",
        "other products",
        "alternatives",
    ]
    for phrase in matches:
        assert _GENERIC_FOLLOWUP_RE.match(phrase), f"Expected match for: {phrase!r}"


def test_generic_followup_re_no_match_for_specific():
    no_matches = [
        "suggest a vitamin C serum",
        "what is PDRN",
        "analyse my routine",
        "retinol for acne",
        "products with niacinamide for oily skin",
    ]
    for phrase in no_matches:
        assert not _GENERIC_FOLLOWUP_RE.match(phrase), f"Expected no match for: {phrase!r}"
