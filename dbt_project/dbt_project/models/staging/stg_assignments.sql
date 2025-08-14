with src as (
  select
    cast(user_id as bigint) as user_id,
    lower(cast(variant as varchar)) as variant,
    cast(assigned_at as timestamp) as assigned_at,
    lower(cast(platform as varchar)) as platform,
    lower(cast(acquisition_channel as varchar)) as acquisition_channel
  from {{ ref('assignments_seed') }}
)

select *
from src
