import airportsdata

# airportsdata API 버전 차이를 흡수
try:
    data = airportsdata.load("IATA")
except Exception:
    data = airportsdata.load()

codes = sorted([k for k in data.keys() if isinstance(k, str) and len(k) == 3])
# outlines는 AIRPORT_LIST를 import하므로 그 이름만 맞춰주면 됨
AIRPORT_LIST = [{"iata": c, "iata_code": c} for c in codes]
