#include <WiFi.h>
#include <HTTPClient.h>
#include <ArduinoJson.h>
#include <ESP32Servo.h>

// WiFi credentials
char ssid[] = "1234556788";
char pass[] = "sayabukanlah";

// Server URLs
const char *serverurlA1 = "http://192.168.137.1:5000/get_detections";
const char *serverurlA2 = "http://192.168.137.1:5001/get_detections";

// Pin setup
#define SERVO_PIN 2
#define IR_1_PIN 13

Servo servo;
bool gateOpen = false;

void setup()
{
  Serial.begin(115200);
  WiFi.begin(ssid, pass);
  while (WiFi.status() != WL_CONNECTED)
  {
    delay(250);
    Serial.print(".");
  }

  delay(2000);

  servo.attach(SERVO_PIN);
  pinMode(IR_1_PIN, INPUT);

  Serial.println("Setup completed...");
}

void loop()
{
  bool irDetected = digitalRead(IR_1_PIN) == LOW; // LOW = terhalang
  Serial.print("IR: ");
  Serial.println(irDetected ? "DETECTED" : "CLEAR");

  if (irDetected)
  {
    int carA1 = getCarCount(serverurlA1);
    int carA2 = getCarCount(serverurlA2);

    int sisaA1 = 4 - carA1;
    int sisaA2 = 4 - carA2;

    Serial.printf("Sisa A1: %d | Sisa A2: %d\n", sisaA1, sisaA2);

    if ((sisaA1 > 0) || (sisaA2 > 0))
    {
      handleGate(true); // Buka gerbang
    }
    else
    {
      Serial.println("Area parkir penuh, tidak buka gerbang.");
    }
  }
  else
  {
    if (gateOpen)
    {
      delay(1500);
      handleGate(false); // Tutup gerbang
    }
  }

  delay(1000); // Jeda loop
}

int getCarCount(const char *url)
{
  HTTPClient http;
  int carCount = 0;

  if ((WiFi.status() == WL_CONNECTED))
  {
    http.begin(url);
    int httpCode = http.GET();

    if (httpCode > 0)
    {
      String payload = http.getString();
      Serial.println("Response: " + payload);

      StaticJsonDocument<256> doc;
      DeserializationError error = deserializeJson(doc, payload);

      if (!error)
      {
        carCount = doc["object_counts"]["car"];
      }
      else
      {
        Serial.println("JSON parse error");
      }
    }
    else
    {
      Serial.println("HTTP error: " + http.errorToString(httpCode));
    }

    http.end();
  }

  return carCount;
}

void handleGate(bool isOpen)
{
  if (isOpen && !gateOpen)
  {
    Serial.println("Opening gate...");
    for (int angle = 0; angle <= 90; angle++)
    {
      servo.write(angle);
      delay(15);
    }
    gateOpen = true;
  }
  else if (!isOpen && gateOpen)
  {
    Serial.println("Closing gate...");
    for (int angle = 90; angle >= 0; angle--)
    {
      servo.write(angle);
      delay(15);
    }
    gateOpen = false;
  }
}
