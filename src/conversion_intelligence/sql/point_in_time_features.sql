WITH eligible_page_events AS (
    SELECT
        score_requests.score_id,
        page_events.event_at_us,
        page_events.page_category
    FROM score_requests
    INNER JOIN page_events
        ON page_events.session_id = score_requests.session_id
    WHERE page_events.event_at_us < score_requests.score_at_us
      AND page_events.available_at_us <= score_requests.score_at_us
),
page_features AS (
    SELECT
        score_id,
        COUNT(*) AS pages_observed_at_score_time,
        COUNT(DISTINCT page_category) AS unique_page_categories_observed,
        MAX(event_at_us) AS feature_max_event_at_us
    FROM eligible_page_events
    GROUP BY score_id
)
SELECT
    score_requests.score_id,
    sessions.session_id,
    sessions.user_id,
    score_requests.score_at_us,
    sessions.country,
    sessions.age,
    sessions.new_user,
    sessions.source,
    CAST(
        (score_requests.score_at_us - sessions.session_start_at_us) / 1000000
        AS BIGINT
    ) AS seconds_since_session_start,
    COALESCE(page_features.pages_observed_at_score_time, 0)
        AS pages_observed_at_score_time,
    COALESCE(page_features.unique_page_categories_observed, 0)
        AS unique_page_categories_observed,
    page_features.feature_max_event_at_us,
    CASE
        WHEN page_features.feature_max_event_at_us IS NULL THEN NULL
        ELSE CAST(
            (score_requests.score_at_us - page_features.feature_max_event_at_us) / 1000000
            AS BIGINT
        )
    END AS seconds_since_last_page_view
FROM score_requests
INNER JOIN sessions
    ON sessions.session_id = score_requests.session_id
LEFT JOIN page_features
    ON page_features.score_id = score_requests.score_id
