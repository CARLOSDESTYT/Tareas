import requests
import random
import time

API_KEY = "TU_WRITE_API_KEY"

while True:
    # Simular potenciómetro
    valores = [random.uniform(0, 100) for _ in range(20)]

    promedio = sum(valores) / len(valores)
    maximo = max(valores)
    minimo = min(valores)

    # Simular botón
    contador = random.randint(0, 50)
    doble = random.randint(0, 10)

    url = f"https://api.thingspeak.com/update?api_key={API_KEY}" \
          f"&field1={promedio}" \
          f"&field2={maximo}" \
          f"&field3={minimo}" \
          f"&field5={contador}" \
          f"&field6={doble}"

    response = requests.get(url)

    print("Enviado:", response.text)

    time.sleep(20)