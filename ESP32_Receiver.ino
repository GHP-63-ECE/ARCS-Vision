#include "BluetoothSerial.h"

// Initialize Bluetooth Serial 
BluetoothSerial SerialBT;

void setup() {
  // Initialize onboard LED as data-activity tracker
  pinMode(2, OUTPUT);
  digitalWrite(2, LOW);

  // Open the wireless Bluetooth serial channel
  SerialBT.begin("esp32devtest1"); 
}

void loop() {
  // Check if data has arrived wirelessly from the Mac
  if (SerialBT.available() > 0) {
    digitalWrite(2, HIGH); // Flash the built-in LED
    
    // Read incoming string up to the newline terminator
    String rawData = SerialBT.readStringUntil('\n');
    rawData.trim();
    
    if (rawData.length() > 0) {
      // Echo the exact raw JSON string back to the Mac wirelessly
      SerialBT.println(rawData);
    }
    
    digitalWrite(2, LOW);
  }
}
