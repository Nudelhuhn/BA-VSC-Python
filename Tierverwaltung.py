class Tierverwaltung:
    def __init__(selbst):
        selbst.tiere = [] #Leere Liste

    def tier_hinzufügen(selbst, tier):
        selbst.tiere.append(tier)

    def alle_Tiere_anzeigen(selbst):
        for tier in selbst.tiere:
            print(tier.sprich())
