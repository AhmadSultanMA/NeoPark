#include "esp_camera.h"
#include <WiFi.h>
#include <HTTPClient.h>

// WiFi credentials
const char* ssid = "1234556788";
const char* password = "sayabukanlah";

// Server URLs - change port for different areas
const char* serverUrl = "http://192.168.137.1:5000/upload"; // For Area A1
// const char* serverUrl = "http://192.168.137.1:5001/upload"; // For Area A2

void setup() {
  Serial.begin(115200);
  Serial.println("Starting ESP32-CAM...");

  // Camera configuration for AI-EYE ESP32-CAM
  camera_config_t config;
  config.ledc_channel = LEDC_CHANNEL_0;
  config.ledc_timer = LEDC_TIMER_0;
  config.pin_d0 = 5;
  config.pin_d1 = 18;
  config.pin_d2 = 19;
  config.pin_d3 = 21;
  config.pin_d4 = 36;
  config.pin_d5 = 39;
  config.pin_d6 = 34;
  config.pin_d7 = 35;
  config.pin_xclk = 0;
  config.pin_pclk = 22;
  config.pin_vsync = 25;
  config.pin_href = 23;
  config.pin_sscb_sda = 26;
  config.pin_sscb_scl = 27;
  config.pin_pwdn = 32;
  config.pin_reset = -1;
  config.xclk_freq_hz = 20000000;
  config.pixel_format = PIXFORMAT_JPEG;

  // Optimize for better quality and performance
  if(psramFound()){
    config.frame_size = FRAMESIZE_SVGA; // 800x600 for better detection
    config.jpeg_quality = 8; // Better quality for YOLO
    config.fb_count = 2;
    Serial.println("PSRAM found - using higher resolution");
  } else {
    config.frame_size = FRAMESIZE_VGA; // 640x480
    config.jpeg_quality = 10;
    config.fb_count = 1;
    Serial.println("No PSRAM - using standard resolution");
  }

  // Initialize Camera
  esp_err_t err = esp_camera_init(&config);
  if (err != ESP_OK) {
    Serial.printf("Camera init failed with error 0x%x\n", err);
    return;
  }
  
  Serial.println("Camera initialized successfully");

  // Connect to WiFi
  WiFi.begin(ssid, password);
  Serial.print("Connecting to WiFi");
  
  int attempts = 0;
  while(WiFi.status() != WL_CONNECTED && attempts < 30){
    delay(500);
    Serial.print(".");
    attempts++;
  }
  
  if(WiFi.status() == WL_CONNECTED) {
    Serial.println("\nWiFi connected successfully!");
    Serial.print("IP address: ");
    Serial.println(WiFi.localIP());
    Serial.print("Signal strength: ");
    Serial.println(WiFi.RSSI());
  } else {
    Serial.println("\nFailed to connect to WiFi!");
    return;
  }
  
  // Camera sensor settings for better detection
  sensor_t * s = esp_camera_sensor_get();
  if (s != NULL) {
    // Adjust camera settings for better object detection
    s->set_brightness(s, 0);     // -2 to 2
    s->set_contrast(s, 0);       // -2 to 2
    s->set_saturation(s, 0);     // -2 to 2
    s->set_special_effect(s, 0); // 0 to 6 (0 = No Effect)
    s->set_whitebal(s, 1);       // 0 = disable, 1 = enable
    s->set_awb_gain(s, 1);       // 0 = disable, 1 = enable
    s->set_wb_mode(s, 0);        // 0 to 4
    s->set_exposure_ctrl(s, 1);  // 0 = disable, 1 = enable
    s->set_aec2(s, 0);           // 0 = disable, 1 = enable
    s->set_ae_level(s, 0);       // -2 to 2
    s->set_aec_value(s, 300);    // 0 to 1200
    s->set_gain_ctrl(s, 1);      // 0 = disable, 1 = enable
    s->set_agc_gain(s, 0);       // 0 to 30
    s->set_gainceiling(s, (gainceiling_t)0); // 0 to 6
    s->set_bpc(s, 0);            // 0 = disable, 1 = enable
    s->set_wpc(s, 1);            // 0 = disable, 1 = enable
    s->set_raw_gma(s, 1);        // 0 = disable, 1 = enable
    s->set_lenc(s, 1);           // 0 = disable, 1 = enable
    s->set_hmirror(s, 0);        // 0 = disable, 1 = enable
    s->set_vflip(s, 0);          // 0 = disable, 1 = enable
    s->set_dcw(s, 1);            // 0 = disable, 1 = enable
    s->set_colorbar(s, 0);       // 0 = disable, 1 = enable
    
    Serial.println("Camera settings optimized for object detection");
  }
  
  Serial.println("Setup complete. Starting image capture loop...");
}

void loop() {
  // Check WiFi connection
  if(WiFi.status() != WL_CONNECTED) {
    Serial.println("WiFi disconnected. Attempting to reconnect...");
    WiFi.reconnect();
    delay(5000);
    return;
  }
  
  // Capture image
  camera_fb_t * fb = esp_camera_fb_get();
  if (!fb) {
    Serial.println("Camera capture failed");
    delay(100);
    return;
  }
  
  Serial.printf("Captured image: %zu bytes\n", fb->len);
  
  // Send image to server
  WiFiClient client;
  HTTPClient http;
  
  http.begin(client, serverUrl);
  http.addHeader("Content-Type", "image/jpeg");
  http.addHeader("Content-Length", String(fb->len));
  http.setTimeout(10000); // 10 second timeout
  
  int httpResponseCode = http.POST(fb->buf, fb->len);
  
  if (httpResponseCode > 0) {
    String response = http.getString();
    Serial.printf("HTTP Response: %d\n", httpResponseCode);
    Serial.println("Response: " + response);
    
    if (httpResponseCode == 200) {
      Serial.println("Image uploaded successfully");
    }
  } else {
    Serial.printf("Failed to upload image, error: %s\n", 
                  http.errorToString(httpResponseCode).c_str());
  }

  http.end();
  esp_camera_fb_return(fb);
  
  // Print memory status
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  
  // Wait before next capture (adjust as needed)
  delay(1000); // 1 second interval for smoother video
}