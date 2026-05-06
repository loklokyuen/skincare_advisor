"""
conflict_lookup.py

Curated lookup table of known ingredient conflicts.
Every pair is bidirectional — if A conflicts with B, B conflicts with A.

Sources used to compile this list:
  - INCIDecoder ingredient notes (incidecoder.com)
  - Paula's Choice ingredient dictionary (paulaschoice.com/ingredient-dictionary)
  - CosDNA ingredient flags (cosdna.com)
  - Dermatology consensus and cosmetic chemistry literature

Conflict levels:
  "avoid"   — combination is likely to cause irritation, reduce efficacy, or
               cause a chemical reaction. Do not use together.
  "caution" — combination can work but needs care: separate AM/PM, buffer
               with moisturiser, or lower concentration of one or both.

All INCI names are normalised: lowercase, no punctuation except hyphens.

To add a new pair:
  Add one entry to CONFLICT_PAIRS. The build_index() call produces the
  bidirectional lookup automatically — you never need to add both directions.
"""

# ---------------------------------------------------------------------------
# Conflict pairs — add new entries here only
# Each tuple: (inci_a, inci_b, level, reason)
# ---------------------------------------------------------------------------

CONFLICT_PAIRS: list[tuple[str, str, str, str]] = [

    # Retinoids vs exfoliating acids
    # Using both together significantly raises irritation risk. AHAs lower
    # skin pH which can also destabilise retinol.
    ("retinol",          "glycolic acid",   "avoid",
     "Combined use significantly increases irritation and dryness. "
     "Use on alternate evenings or separate AM/PM."),

    ("retinol",          "lactic acid",     "avoid",
     "Combined use significantly increases irritation and dryness. "
     "Use on alternate evenings or separate AM/PM."),

    ("retinol",          "salicylic acid",  "avoid",
     "Combined use significantly increases irritation and peeling. "
     "Use on alternate evenings or separate AM/PM."),

    ("retinol",          "mandelic acid",   "avoid",
     "Combined use raises irritation risk. "
     "Use on alternate evenings or separate AM/PM."),

    ("retinol",          "azelaic acid",    "caution",
     "Both are active at low pH. Start by separating AM/PM and monitor "
     "for irritation before combining in the same routine step."),

    ("retinal",          "glycolic acid",   "avoid",
     "Retinal is more potent than retinol. Combined use with AHAs "
     "significantly increases irritation risk. Use on alternate evenings."),

    ("retinal",          "lactic acid",     "avoid",
     "Retinal is more potent than retinol. Combined use with AHAs "
     "significantly increases irritation risk. Use on alternate evenings."),

    ("retinal",          "salicylic acid",  "avoid",
     "Retinal is more potent than retinol. Combined use with BHA "
     "significantly increases irritation risk. Use on alternate evenings."),

    # Retinoids vs vitamin C
    # Vitamin C (ascorbic acid) is most stable and effective at low pH (~3.5).
    # Retinol is destabilised at that pH. They also both cause irritation
    # individually so combining is high risk for sensitive skin.
    ("retinol",          "ascorbic acid",   "avoid",
     "Ascorbic acid works at low pH which destabilises retinol. "
     "Both are individually sensitising. Use vitamin C in AM, retinol in PM."),

    ("retinal",          "ascorbic acid",   "avoid",
     "Ascorbic acid works at low pH which destabilises retinoids. "
     "Use vitamin C in AM, retinal in PM."),

    # Retinoids vs benzoyl peroxide
    # BPO oxidises retinol on contact, rendering it ineffective.
    ("retinol",          "benzoyl peroxide", "avoid",
     "Benzoyl peroxide oxidises retinol on contact, making it ineffective. "
     "Use on separate evenings or apply BPO in AM only."),

    ("retinal",          "benzoyl peroxide", "avoid",
     "Benzoyl peroxide oxidises retinoids on contact, making them ineffective. "
     "Use on separate evenings or apply BPO in AM only."),

    ("retinyl palmitate", "benzoyl peroxide", "avoid",
     "Benzoyl peroxide oxidises retinoids. Use on separate occasions."),

    # Vitamin C (ascorbic acid) vs niacinamide
    # At high concentrations both together can form nicotinic acid (niacin),
    # causing flushing and yellowing of the formula. At typical cosmetic
    # concentrations (<10% each) the risk is low but still worth noting.
    ("ascorbic acid",    "niacinamide",     "caution",
     "At high concentrations both can form nicotinic acid, causing skin "
     "flushing and formula yellowing. At typical cosmetic concentrations "
     "(<10% each) the risk is low. Separate AM/PM if using high-strength "
     "versions of both."),

    # AHAs vs BHAs combined at high strength
    # Using multiple exfoliating acids together at active concentrations
    # significantly raises over-exfoliation and barrier damage risk.
    ("glycolic acid",    "salicylic acid",  "caution",
     "Using both at active concentrations risks over-exfoliation and barrier "
     "damage. Use on alternate days or use one as a toner at low concentration "
     "alongside the other."),

    ("lactic acid",      "salicylic acid",  "caution",
     "Using both at active concentrations risks over-exfoliation and barrier "
     "damage. Use on alternate days or use one at low concentration."),

    ("glycolic acid",    "lactic acid",     "caution",
     "Combining two AHAs at active concentrations increases irritation and "
     "over-exfoliation risk. Choose one as your primary exfoliant."),

    # Vitamin C instability with alkaline or high-pH ingredients
    # Ascorbic acid is most stable and effective at pH ~3.5. Ingredients
    # with high pH (alkaline cleansers, sodium bicarbonate) or formulated
    # at high pH denature it quickly.
    ("ascorbic acid",    "sodium bicarbonate", "avoid",
     "Sodium bicarbonate is strongly alkaline and rapidly degrades ascorbic "
     "acid. Do not combine."),

    # Benzoyl peroxide vs vitamin C
    ("benzoyl peroxide", "ascorbic acid",   "avoid",
     "Benzoyl peroxide oxidises ascorbic acid, making it ineffective and "
     "potentially causing discolouration. Use BPO in AM and vitamin C in PM "
     "or on separate days."),

    # Copper peptides vs vitamin C and AHAs
    # Ascorbic acid chelates copper ions, reducing the efficacy of copper
    # peptides. AHAs can also destabilise peptide complexes at low pH.
    ("copper tripeptide-1", "ascorbic acid",  "caution",
     "Ascorbic acid chelates copper ions, reducing the efficacy of copper "
     "peptides. Use vitamin C in AM and copper peptides in PM."),

    ("copper tripeptide-1", "glycolic acid",  "caution",
     "Low pH from AHAs can destabilise copper peptide complexes. "
     "Separate AM/PM or use on alternate days."),

    ("copper tripeptide-1", "lactic acid",    "caution",
     "Low pH from AHAs can destabilise copper peptide complexes. "
     "Separate AM/PM or use on alternate days."),

    # Benzoyl peroxide vs most actives
    # BPO is a strong oxidiser and degrades many actives on contact.
    ("benzoyl peroxide", "niacinamide",     "caution",
     "Benzoyl peroxide can oxidise and degrade niacinamide. Use BPO in AM "
     "and niacinamide in PM, or use different products for each step."),

    # Salicylic acid vs certain peptides
    # Very low pH from BHA can hydrolyse peptide bonds over time.
    ("salicylic acid",   "palmitoyl tripeptide-1",  "caution",
     "Low pH from salicylic acid can hydrolyse peptide bonds over time, "
     "reducing peptide efficacy. Separate AM/PM."),

    ("salicylic acid",   "palmitoyl tetrapeptide-7", "caution",
     "Low pH from salicylic acid can hydrolyse peptide bonds over time. "
     "Separate AM/PM."),

    ("salicylic acid",   "acetyl hexapeptide-3",    "caution",
     "Low pH from salicylic acid can hydrolyse peptide bonds over time. "
     "Separate AM/PM."),

    # Kojic acid instability
    # Kojic acid oxidises and turns brown in the presence of iron ions
    # and is also destabilised by high pH.
    ("kojic acid",       "ascorbic acid",   "caution",
     "Both are unstable and can oxidise each other when combined in high "
     "concentrations. Separate AM/PM or use a formulation that stabilises both."),

    # Enzyme exfoliants vs AHAs
    # Combining enzyme exfoliants (papain, bromelain) with acid exfoliants
    # risks significant over-exfoliation.
    ("papain",           "glycolic acid",   "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation and barrier damage risk. Use on separate days."),

    ("papain",           "lactic acid",     "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation risk. Use on separate days."),

    ("papain",           "salicylic acid",  "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation risk. Use on separate days."),

    ("bromelain",        "glycolic acid",   "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation and barrier damage risk. Use on separate days."),

    ("bromelain",        "lactic acid",     "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation risk. Use on separate days."),

    ("bromelain",        "salicylic acid",  "avoid",
     "Combining enzyme and acid exfoliants significantly increases "
     "over-exfoliation risk. Use on separate days."),
]


# ---------------------------------------------------------------------------
# Index builder — call once, use everywhere
# ---------------------------------------------------------------------------

def build_index() -> dict[str, list[dict]]:
    """
    Returns a dict keyed by INCI name.
    Each value is a list of conflict objects:
      {
        "inci_name": str,   # the other ingredient
        "level":     str,   # "avoid" or "caution"
        "reason":    str,   # plain English explanation
      }
    Every pair is stored in both directions automatically.
    """
    index: dict[str, list[dict]] = {}

    for a, b, level, reason in CONFLICT_PAIRS:
        a = a.lower().strip()
        b = b.lower().strip()

        if a not in index:
            index[a] = []
        if b not in index:
            index[b] = []

        index[a].append({"inci_name": b, "level": level, "reason": reason})
        index[b].append({"inci_name": a, "level": level, "reason": reason})

    return index


# Singleton — import and use CONFLICT_INDEX directly
CONFLICT_INDEX: dict[str, list[dict]] = build_index()


def get_conflicts(inci_name: str) -> list[dict]:
    """
    Returns the conflict list for a given INCI name.
    Returns empty list if no conflicts are recorded.
    """
    return CONFLICT_INDEX.get(inci_name.lower().strip(), [])


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    print(f"Total ingredients with recorded conflicts: {len(CONFLICT_INDEX)}\n")

    # Show a few examples
    for name in ["retinol", "ascorbic acid", "glycolic acid", "niacinamide", "copper tripeptide-1"]:
        conflicts = get_conflicts(name)
        print(f"{name} ({len(conflicts)} conflict(s)):")
        for c in conflicts:
            print(f"  [{c['level'].upper()}] {c['inci_name']}")
            print(f"    {c['reason'][:80]}...")
        print()