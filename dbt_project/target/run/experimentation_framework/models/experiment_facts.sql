
  
  create view "local"."analytics"."experiment_facts__dbt_tmp" as (
    with a as ( select * from "local"."analytics"."assignments"), 
    o as ( select * from "local"."analytics"."outcomes" )
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
  );
