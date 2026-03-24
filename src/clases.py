class SensorData:
    def __init__(self, reactor_temp, reactor_pressure, vibration, ambient_temp_effect, feed_flow_rate):
        self.reactor_temp = reactor_temp
        self.reactor_pressure = reactor_pressure
        self.vibratin = vibration
        self.ambient_temp_effect = ambient_temp_effect
        self.feed_flow_rate = feed_flow_rate 