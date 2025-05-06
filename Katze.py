from Tier import Tier

class Katze(Tier):
    def __init__(selbst, name, alter, farbe):
        super().__init__(name, alter)
        selbst.farbe = farbe

    def sprich(selbst):
        return f"Miau! Ich bin {selbst.name}, eine Katze mit der Farbe {selbst.farbe}"