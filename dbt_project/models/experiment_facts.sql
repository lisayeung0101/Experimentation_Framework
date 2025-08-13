with a as ( select * from {{ ref('assignments') }}), 
    o as ( select * from {{ ref('outcomes') }} )
select
    a.user_id,
    a.experiment_id,
    a.variant,
    o.conversion,
    o.revenue,
    o.pre_metric,
    o.event_ts
from a
left join o using (user_id, experiment_id)