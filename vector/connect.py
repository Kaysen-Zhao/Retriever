import requests
from utiles import get_global_config


def access_vector_api(data):
    url = "http://%s:%s%s" % (get_global_config()['vector']['api']['host'], get_global_config()[
        'vector']['api']['port'], get_global_config()['vector']['api']['path'])
    headers = {
        "Content-Type": "application/json; charset=UTF-8",
        "Authorization": "Bearer " + get_global_config()['vector']['api']['token']
    }
    res = requests.post(url, headers=headers, json=data)
    if res.status_code == 200:
        return res.json()
    else:
        return {'error': res.text, 'success': False}
