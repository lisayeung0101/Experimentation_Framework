select cast(user_id as string) as user_id,
cast(experiment_id as string) as experiment_id,
cast(variant as string) as variant,
assigned_at
from "local"."analytics_analytics"."assignments_seed"