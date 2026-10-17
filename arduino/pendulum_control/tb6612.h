#ifndef TB6612_H
#define TB6612_H

#include <Arduino.h>
#include "config.h"

class TB6612 {
public:
  TB6612();
  
  void begin();
  void setForce(float force_n);  // Сила в Ньютонах (-MAX_FORCE_N ... +MAX_FORCE_N)
  void enable();
  void disable();
  void brake();
  
  float getCurrentForce() const { return _current_force; }
  bool isEnabled() const { return _enabled; }

private:
  float _current_force = 0;
  bool _enabled = false;
  
  // Преобразование силы в PWM
  uint8_t forceToPWM(float force_n) const;
};

#endif // TB6612_H