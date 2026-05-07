from typing import Annotated, Literal, Optional, TypedDict, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class UserProfile(TypedDict):
    user_id: str
    name: str
    age_range: Optional[str]
    skin_type: str
    concerns: list[str]
    goals: list[str]
    notes: NotRequired[str]


TimeOfDay = Literal["AM", "PM"]
RoutineType = Literal["active", "rest"]
RoutineStep = Literal[
    "cleanser",
    "sunscreen",
    "toner",
    "serum",
    "moisturiser",
    "cream",
    "mask",
    "exfoliant",
    "treatment",
]


class RoutineProduct(TypedDict):
    product_id: int
    product_name: str
    brand: Optional[str]
    time_of_day: TimeOfDay
    step: RoutineStep
    type: RoutineType
    days_of_week: NotRequired[list[int]]
    ingredients: list[str]


class UserRoutine(TypedDict):
    user_id: str
    active: dict[TimeOfDay, list[RoutineProduct]]
    rest: dict[TimeOfDay, list[RoutineProduct]]


class Product(TypedDict):
    product_id: int
    product_name: str
    brand: Optional[str]
    quantity: NotRequired[str]
    image_url: NotRequired[str]
    ingredients: list[str]
    key_ingredients: NotRequired[list[str]]
    active_ingredients: NotRequired[list[str]]
    categories: NotRequired[list[str]]
    source: str


class ProductSearchResult(TypedDict):
    """Shape returned by product_service.search_products — UI-facing."""
    product_name: str
    brand: str
    quantity: NotRequired[str]
    image_url: str
    key_ingredients: list[str]
    active_ingredients: NotRequired[list[str]]
    ingredients: NotRequired[list[str]]
    ingredients_raw: NotRequired[str]
    categories: list[str]
    source: NotRequired[str]


class IngredientDetail(TypedDict):
    ingredient_id: int
    inci_name: str
    display_name: str           # same as inci_name (no separate column in DB)
    common_names: list[str]
    category: Optional[str]     # not stored; always ""
    functions: list[str]
    cautions: list[str]         # derived from safety_notes + known_conflicts
    suitable_for: list[str]     # maps to suitable_skin_types
    avoid_for: list[str]        # maps to avoid_skin_types
    usage_guidance: NotRequired[str]
    conditions: NotRequired[list[str]]


RiskLevel = Literal["low", "medium", "high", "unknown"]


class ConflictResult(TypedDict):
    risk_level: RiskLevel
    products_involved: list[str]
    ingredients_involved: list[str]
    reason: str
    suggestion: str


class AnalysisResults(TypedDict):
    conflicts: list[ConflictResult]
    gaps: NotRequired[list[str]]
    suitability_notes: NotRequired[list[str]]


Mode = Literal["build", "analyse", "learn", "recommend"]


class GraphState(TypedDict):
    user_id: str
    mode: Mode
    mode_locked: NotRequired[bool]
    message: str
    messages: Annotated[list[BaseMessage], add_messages]
    user_profile: UserProfile
    user_routine: UserRoutine
    retrieved_products: list[Product]
    retrieved_ingredients: list[IngredientDetail]
    matched_products: NotRequired[list[ProductSearchResult]]  # full results with image_url shown as cards
    analysis_results: NotRequired[AnalysisResults]
    input_status: NotRequired[dict]
    tool_outputs: NotRequired[list[dict]]
    draft_response: NotRequired[str]
    final_response: NotRequired[str]
    advisor_notes: NotRequired[str]
    product_recommendation_analysis: NotRequired[list[dict]]
    ingredient_recommendation_analysis: NotRequired[list[dict]]
    agent_selected_product_names: NotRequired[list[str]]
    community_search_request: NotRequired[dict]
    literature_search_request: NotRequired[dict]
    evidence_summary: NotRequired[dict]
    context_block: NotRequired[str]
    ingredient_terms: NotRequired[list[str]]
