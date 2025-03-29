import http.client
import json

conn = http.client.HTTPSConnection("api.chatanywhere.tech")
payload = ''
headers = {
   'Authorization': 'Bearer sk-dqbCjalqSxgKaqe4YyNGGByaNLFk6vv0gXp0LnErebFmTZkx'
}
conn.request("GET", "/v1/models", payload, headers)
res = conn.getresponse()
data = res.read()

# Parse the JSON response
response = json.loads(data.decode("utf-8"))

# Extract unique model IDs
model_ids = []
if 'data' in response:
    unique_models = set()
    for model in response['data']:
        if 'id' in model and model['id'] not in unique_models:
            unique_models.add(model['id'])
            # model_ids.append({
            #     'id': model['id'],
            #     'owner': model.get('owned_by', 'unknown')
            # })

print(unique_models)
