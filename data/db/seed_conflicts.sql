-- Run once against your Cloud SQL instance to create and seed the conflict rules table.

CREATE TABLE IF NOT EXISTS skincare_advisor.ingredient_conflicts (
    id            SERIAL PRIMARY KEY,
    ingredient_a  VARCHAR(200) NOT NULL,
    ingredient_b  VARCHAR(200) NOT NULL,
    risk_level    VARCHAR(20)  NOT NULL CHECK (risk_level IN ('low', 'medium', 'high')),
    reason        TEXT         NOT NULL,
    suggestion    TEXT         NOT NULL
);

CREATE UNIQUE INDEX IF NOT EXISTS uq_conflict_pair
    ON skincare_advisor.ingredient_conflicts (
        LEAST(ingredient_a, ingredient_b),
        GREATEST(ingredient_a, ingredient_b)
    );

INSERT INTO skincare_advisor.ingredient_conflicts
    (ingredient_a, ingredient_b, risk_level, reason, suggestion)
VALUES
    ('retinol', 'benzoyl peroxide', 'high',
     'Benzoyl peroxide can oxidise retinoids and increase irritation.',
     'Separate into different routines or alternate nights.'),

    ('retinol', 'glycolic acid', 'medium',
     'Retinoids plus strong AHAs can cause dryness and barrier disruption.',
     'Alternate nights; use AHA in the AM and retinol at night.'),

    ('retinol', 'salicylic acid', 'medium',
     'Retinoids plus BHA can be too drying, especially on sensitive skin.',
     'Introduce slowly; separate use if sensitivity appears.'),

    ('retinol', 'lactic acid', 'medium',
     'Lactic acid lowers skin pH, which can increase retinol irritation.',
     'Apply on alternate nights or use lactic acid in the AM routine.'),

    ('retinol', 'azelaic acid', 'low',
     'Both are active ingredients; layering may cause unnecessary irritation for beginners.',
     'Fine for experienced users — start with alternating nights if new to either.'),

    ('vitamin c', 'glycolic acid', 'medium',
     'Both are low-pH; layering them can sting and irritate sensitised skin.',
     'Use vitamin C in the AM and glycolic acid in the PM.'),

    ('vitamin c', 'benzoyl peroxide', 'medium',
     'Benzoyl peroxide can oxidise and inactivate vitamin C (ascorbic acid).',
     'Use vitamin C in the AM and benzoyl peroxide in the PM.'),

    ('niacinamide', 'vitamin c', 'low',
     'At high concentrations both can form niacin, causing temporary flushing. Unlikely at typical cosmetic doses.',
     'Generally safe to layer — stagger by a few minutes if you notice any redness.'),

    ('glycolic acid', 'salicylic acid', 'low',
     'Combining two exfoliating acids increases over-exfoliation risk.',
     'Use one per session; alternate or keep them in separate AM/PM slots.'),

    ('retinol', 'retinal', 'medium',
     'Retinal is significantly more potent than retinol — using both is excessive and irritating.',
     'Choose one retinoid. Retinal converts faster; use it alone at a lower frequency.')

ON CONFLICT DO NOTHING;
