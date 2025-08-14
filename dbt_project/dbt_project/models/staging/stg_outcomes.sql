with src as (
  select
    cast(user_id as bigint) as user_id,
    cast(event_date as date) as event_date,
    cast(trial_start as boolean) as trial_start,
    cast(nullif(trial_start_at, '') as timestamp) as trial_start_at,
    cast(paid_subscriber as boolean) as paid_subscriber,
    cast(nullif(paid_at, '') as timestamp) as paid_at,
    cast(refund_in_first_cycle as boolean) as refund_in_first_cycle,
    cast(early_churn_30d as boolean) as early_churn_30d,
    cast(time_to_subscribe_days as double) as time_to_subscribe_days,
    cast(pre_engagement_30d as integer) as pre_engagement_30d
  from {{ ref('outcomes_seed') }}
)

select *
from src
