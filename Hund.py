from Tier import Tier

class Hund(Tier):
    def __init__(selbst, name, alter, rasse):
        super().__init__(name, alter)
        selbst.rasse = rasse

    def sprich(selbst):
        return f"Wuff! Ich heiße {selbst.name} und bin ein {selbst.rasse}"
