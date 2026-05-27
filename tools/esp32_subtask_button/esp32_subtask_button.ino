// Subtask button box for lerobot-record.
//
// Wire each button between a GPIO and GND.  INPUT_PULLUP is used, so a press
// pulls the pin LOW.  On a debounced press, the sketch prints "BTN <id>\n"
// over USB CDC serial, where <id> is the index into BUTTON_PINS (1-based).
//
// Pin map (button id -> GPIO):
//   1 -> 27
//   2 -> 26
//   3 -> 14
//   4 -> 25
//   5 -> 13
//   6 -> 12
//
// Host-side config (SubtaskAnnotatorConfig) maps button id -> function
// (next / back / stop / skip_forward / skip_backward).

#include <Arduino.h>

const uint8_t BUTTON_PINS[] = {27, 26, 14, 25, 13, 12};
const uint8_t NUM_BUTTONS = sizeof(BUTTON_PINS) / sizeof(BUTTON_PINS[0]);
const unsigned long DEBOUNCE_MS = 20;

uint8_t lastStable[NUM_BUTTONS];
uint8_t lastReading[NUM_BUTTONS];
unsigned long lastChangeMs[NUM_BUTTONS];

// Heartbeat for the host-side find_port helper: emit ID line every second.
const unsigned long HEARTBEAT_MS = 1000;
unsigned long lastHeartbeatMs = 0;

void setup() {
  Serial.begin(115200);
  Serial.println("ID:BUTTONS");
  for (uint8_t i = 0; i < NUM_BUTTONS; i++) {
    pinMode(BUTTON_PINS[i], INPUT_PULLUP);
    lastStable[i] = HIGH;
    lastReading[i] = HIGH;
    lastChangeMs[i] = 0;
  }
}

void loop() {
  unsigned long now = millis();
  if (now - lastHeartbeatMs >= HEARTBEAT_MS) {
    Serial.println("ID:BUTTONS");
    lastHeartbeatMs = now;
  }
  for (uint8_t i = 0; i < NUM_BUTTONS; i++) {
    uint8_t reading = digitalRead(BUTTON_PINS[i]);
    if (reading != lastReading[i]) {
      lastChangeMs[i] = now;
      lastReading[i] = reading;
    }
    if ((now - lastChangeMs[i]) > DEBOUNCE_MS && reading != lastStable[i]) {
      lastStable[i] = reading;
      if (reading == LOW) {
        Serial.print("BTN ");
        Serial.println(i + 1);
      }
    }
  }
}
