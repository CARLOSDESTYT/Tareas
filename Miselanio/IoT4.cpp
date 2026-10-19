#include <WiFi.h>
#include "ThingSpeak.h"

// 🔹 WIFI
const char* ssid = "TU_WIFI";
const char* password = "TU_PASSWORD";

// 🔹 ThingSpeak
WiFiClient client;
unsigned long channelID = TU_CHANNEL_ID;
const char* writeAPIKey = "TU_WRITE_API_KEY";
const char* readAPIKey  = "TU_READ_API_KEY";

// 🔹 Pines
#define POT_PIN 34
#define BUTTON_PIN 14
#define LED_PIN 2

// 🔹 Ventana de tiempo
const unsigned long ventana = 20000;
unsigned long lastSample = 0;
unsigned long lastSend = 0;

// 🔹 Variables potenciómetro
float suma = 0;
float maxVal = 0;
float minVal = 100;
int muestras = 0;

// 🔹 Buffer circular
struct Data {
  float prom;
  float max;
  float min;
};

Data buffer[10];
int bufferIndex = 0;
int bufferCount = 0;

// 🔹 Botón
int contadorBoton = 0;
int dobleClick = 0;
unsigned long lastClickTime = 0;
const int doubleClickDelay = 400;

// 🔹 LED
bool modoParpadeo = false;
unsigned long lastBlink = 0;
int intervalo = 500;
bool estadoLED = false;
String ultimoComando = "";

// 🔹 WIFI
void conectarWiFi() {
  WiFi.begin(ssid, password);
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
  }
}

// 🔹 SETUP
void setup() {
  Serial.begin(115200);

  pinMode(BUTTON_PIN, INPUT_PULLUP);
  pinMode(LED_PIN, OUTPUT);

  conectarWiFi();
  ThingSpeak.begin(client);

  // 🔹 Recuperar contadores (persistencia)
  contadorBoton = ThingSpeak.readIntField(channelID, 5, readAPIKey);
  dobleClick = ThingSpeak.readIntField(channelID, 6, readAPIKey);
}

// 🔹 LOOP
void loop() {

  // 🔸 Reconectar WiFi
  if (WiFi.status() != WL_CONNECTED) {
    conectarWiFi();
  }

  // 🔹 1. Muestreo cada 1 segundo
  if (millis() - lastSample >= 1000) {
    lastSample = millis();

    int valor = analogRead(POT_PIN);
    float porcentaje = (valor / 4095.0) * 100;

    suma += porcentaje;
    muestras++;

    if (porcentaje > maxVal) maxVal = porcentaje;
    if (porcentaje < minVal) minVal = porcentaje;
  }

  // 🔹 2. Botón (detección de clic y doble clic)
  if (digitalRead(BUTTON_PIN) == LOW) {
    unsigned long ahora = millis();

    if (ahora - lastClickTime < doubleClickDelay) {
      dobleClick++;
    } else {
      contadorBoton++;
    }

    lastClickTime = ahora;
    delay(200); // debounce
  }

  // 🔹 3. Control LED desde ThingSpeak (Field 7)
  String comando = ThingSpeak.readStringField(channelID, 7, readAPIKey);

  if (comando != ultimoComando) {

    if (comando == "1") {
      digitalWrite(LED_PIN, HIGH);
      modoParpadeo = false;
      ThingSpeak.setField(8, 1);
    }

    else if (comando == "0") {
      digitalWrite(LED_PIN, LOW);
      modoParpadeo = false;
      ThingSpeak.setField(8, 0);
    }

    else {
      float frecuencia = comando.toFloat();

      if (frecuencia >= 0.1 && frecuencia <= 3) {
        intervalo = (1000 / frecuencia) / 2;
        modoParpadeo = true;
        ThingSpeak.setField(8, frecuencia);
      } else {
        ThingSpeak.setField(8, -1); // error
      }
    }

    ultimoComando = comando;
  }

  // 🔹 Parpadeo LED
  if (modoParpadeo && millis() - lastBlink >= intervalo) {
    lastBlink = millis();
    estadoLED = !estadoLED;
    digitalWrite(LED_PIN, estadoLED);
  }

  // 🔹 4. Envío cada 20 segundos
  if (millis() - lastSend >= ventana) {
    lastSend = millis();

    float promedio = suma / muestras;

    // 🔸 Guardar en buffer si falla WiFi
    if (WiFi.status() != WL_CONNECTED) {
      buffer[bufferIndex] = {promedio, maxVal, minVal};
      bufferIndex = (bufferIndex + 1) % 10;
      if (bufferCount < 10) bufferCount++;
    }

    else {

      // 🔸 Enviar datos actuales
      ThingSpeak.setField(1, promedio);
      ThingSpeak.setField(2, maxVal);
      ThingSpeak.setField(3, minVal);
      ThingSpeak.setField(5, contadorBoton);
      ThingSpeak.setField(6, dobleClick);

      ThingSpeak.writeFields(channelID, writeAPIKey);

      // 🔸 Enviar buffer (histórico)
      for (int i = 0; i < bufferCount; i++) {
        int idx = (bufferIndex + i) % bufferCount;

        String data = "P:" + String(buffer[idx].prom) +
                      ",MAX:" + String(buffer[idx].max) +
                      ",MIN:" + String(buffer[idx].min);

        ThingSpeak.setField(4, data);
        ThingSpeak.writeFields(channelID, writeAPIKey);

        delay(20000); // respetar límite ThingSpeak
      }

      bufferCount = 0;
    }

    // 🔸 Reset
    suma = 0;
    muestras = 0;
    maxVal = 0;
    minVal = 100;
  }
}