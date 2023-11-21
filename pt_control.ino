#include <stdlib.h>
#include <Servo.h>
Servo pan, tilt;
String inputString = "";         // a String to hold incoming data
bool stringComplete = false;  // whether the string is complete
int pos_pan = 93;
int pos_tilt = 70;
void setup() {
  // initialize serial:
  Serial.begin(9600);
  // reserve 200 bytes for the inputString:
  inputString.reserve(200);
  pan.attach(9);
  tilt.attach(10);
}

void loop() {
  // print the string when a newline arrives:
  if (stringComplete) {
    if(inputString.substring(0,3) == "pan"){
      pos_pan = atoi(inputString.substring(3).c_str());
      pan.write(pos_pan);
    }
    if(inputString.substring(0,4) == "tilt"){
      pos_tilt = atoi(inputString.substring(4).c_str());
      tilt.write(pos_tilt);
    }
    else;
      
    
    //Serial.println(inputString);
    // clear the string:
    inputString = "";
    stringComplete = false;
  }
}

/*
  SerialEvent occurs whenever a new data comes in the hardware serial RX. This
  routine is run between each time loop() runs, so using delay inside loop can
  delay response. Multiple bytes of data may be available.
*/
void serialEvent() {
  while (Serial.available()) {
    // get the new byte:
    char inChar = (char)Serial.read();
    // add it to the inputString:
    inputString += inChar;
    // if the incoming character is a newline, set a flag so the main loop can
    // do something about it:
    if (inChar == '\n') {
      stringComplete = true;
    }
  }
}
