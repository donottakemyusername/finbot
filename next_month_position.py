from quant_gt_model import predict_month
from datetime import datetime, timedelta
import json

# 自动计算下个月
today = datetime.now()
next_month_date = (today.replace(day=1) + timedelta(days=32)).replace(day=1)
next_month_str = next_month_date.strftime("%Y-%m")

print(f"\n下个月 {next_month_str} 的仓位建议：\n")
result = predict_month(next_month_str)
print(json.dumps(result, indent=2, ensure_ascii=False))