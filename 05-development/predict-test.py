#!/usr/bin/env python
# coding: utf-8

import requests


url = 'http://localhost:9696/predict'

customer_id = 'xyz-123'
customer = {"job": "retired", "duration": 445, "poutcome": "success"}


response = requests.post(url, json=customer).json()
print(response)

if response['credit'] == True:
    print('this client will get credit %s' % customer_id)
else:
    print('this client will not get credit %s' % customer_id)