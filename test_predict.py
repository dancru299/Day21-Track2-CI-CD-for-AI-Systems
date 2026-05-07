import urllib.request, json
data = json.dumps({'features': [7.4, 0.70, 0.00, 1.9, 0.076, 11.0, 34.0, 0.9978, 3.51, 0.56, 9.4, 0]}).encode()
req = urllib.request.Request('http://localhost:8000/predict', data=data, headers={'Content-Type': 'application/json'}, method='POST')
print(urllib.request.urlopen(req).read().decode())
