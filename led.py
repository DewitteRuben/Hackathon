from gpiozero import LED, Button
from signal import pause

# Initialize LED and Button
led = LED(17)  # GPIO17 (physical pin 11)
button = Button(26)  # GPIO26 (physical pin 37)

print("Program started. Press Ctrl+C to exit.")
print("Press the button on GPIO26 to see a message.")

# Define what happens when button is pressed
def button_pressed():
    print("Button pressed!")

# Set up button callback
button.when_pressed = button_pressed

# Start LED blinking (500ms on, 500ms off)
led.blink(on_time=0.5, off_time=0.5)

# Keep the program running
pause()