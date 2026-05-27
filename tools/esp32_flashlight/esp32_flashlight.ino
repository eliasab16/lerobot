// WS2812B flashlight: hold 18 LEDs at a fixed warm tint.

#include <Adafruit_NeoPixel.h>

#define LED_PIN   5
#define NUM_LEDS  30

#define COLOR_R   12
#define COLOR_G   10
#define COLOR_B   4

Adafruit_NeoPixel strip(NUM_LEDS, LED_PIN, NEO_GRB + NEO_KHZ800);

void setup() {
  strip.begin();
  strip.fill(strip.Color(COLOR_R, COLOR_G, COLOR_B));
  strip.show();
}

void loop() {
  delay(1000);
}
