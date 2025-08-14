select cast(user_id as string) as user_id,
cast(experiment_id as string) as experiment_id,
conversion as conversion,
revenue as revenue,
pre_metric as pre_metric,
event_ts
from "local"."analytics_analytics"."outcomes_seed"