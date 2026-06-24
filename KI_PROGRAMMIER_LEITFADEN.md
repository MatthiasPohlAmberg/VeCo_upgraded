# KI-Programmierleitfaden

Ziel: Projekte in diesem Workspace sollen pragmatisch, uebersichtlich und direkt nutzbar sein. Funktion geht vor Architektur. Keine Verschachtelung ohne echten Nutzen.

## Grundregeln

1. Bestehende Projektstruktur respektieren.
2. Einfachste funktionierende Loesung bauen.
3. Monolithisch starten und monolithisch bleiben, solange es lesbar ist.
4. Keine mittleren/grossen Projektstrukturen erzwingen.
5. Keine Architektur auf Vorrat.
6. Keine unnoetigen Klassen, Services, Repositories oder Framework-Schichten.
7. Code sauber benennen, knapp kommentieren und nachvollziehbar loggen.
8. Code immer auf Aktualitaet, Wartbarkeit und Performance betrachten.
9. Keine Secrets, lokalen Artefakte oder generierten Massendaten versionieren.

## Immer zuerst lesen

- `README.md`
- `setup_venv.ps1`
- `requirements.txt`
- `.gitignore`
- `PROJECT_INSTRUCTIONS.md`, falls vorhanden
- Einstiegspunkt: `main.py`, `app.py`, `tool_name.py` oder `__main__.py`

## Projektstruktur

Standard fuer neue Projekte:

```text
projekt/
├── main.py
├── README.md
├── requirements.txt
├── setup_venv.ps1
├── .gitignore
├── input/      # optional, ignored
├── output/     # optional, ignored
└── logs/       # optional, ignored
```

Regeln:

- Ein klarer Einstiegspunkt.
- Eine Datei ist erlaubt und oft richtig.
- Mehrere Dateien nur bei echter Uebersichtlichkeit.
- Hilfsfunktionen im selben Modul lassen, wenn sie dort gut lesbar bleiben.
- Auslagern erst, wenn eine Datei zu lang, unuebersichtlich oder mehrfach gebraucht wird.
- Keine Package-Struktur nur aus Gewohnheit.
- `pyproject.toml` nur anlegen, wenn das Projekt wirklich als Package/Tool verteilt werden soll.

## Code-Stil

- Funktionen klein, klar benannt und direkt testbar halten.
- `main()` als Einstieg verwenden.
- Ablauf von oben nach unten lesbar machen: Konstanten, Logging, Hilfsfunktionen, Fachfunktionen, `main()`.
- Seiteneffekte sichtbar halten: Datei, Netzwerk, Datenbank, Subprozess.
- Fehler nicht verschlucken. Mit Kontext loggen oder bewusst abbrechen.
- Keine cleveren Konstrukte, wenn einfache Schleifen besser lesbar sind.
- Keine globale Magie. Globale Konstanten ja, global veraenderbarer Zustand nein.

## Aktualitaet und Performance

- Veraltete APIs, Pakete und Patterns ersetzen, wenn es ohne grossen Umbau moeglich ist.
- Dependencies aktuell halten, aber keine unnoetigen Upgrades erzwingen.
- Performance bei jeder Aenderung mitdenken: I/O, Schleifen, Speicher, Netzwerk, Modellaufrufe.
- Offensichtlich langsame oder doppelte Arbeit entfernen.
- Grosse Dateien, Medien, Modelle und API-Aufrufe sparsam laden.
- Erst messen oder begruenden, bevor komplex optimiert wird.
- Lesbarkeit nicht fuer Mikro-Optimierungen opfern.

## Kommentare und Dokumentation im Code

- Code muss durch Namen und Struktur verstaendlich sein.
- Kommentare erklaeren Warum und Kontext, nicht offensichtliches Was.
- Jede nicht-triviale Funktion bekommt eine kurze Docstring.
- Komplexe Annahmen direkt an der Stelle dokumentieren.
- Workarounds mit Grund und Grenze kommentieren.
- Veraltete Kommentare sofort entfernen oder aktualisieren.

Beispiel:

```python
def normalize_filename(name: str) -> str:
    """Bereinigt Dateinamen fuer Windows-kompatible Ausgaben."""
    ...
```

## Logging

Jedes Projekt mit Laufzeitlogik soll Logging haben.

Pflicht:

- Logging direkt am Programmeinstieg initialisieren.
- Log-Level konfigurierbar machen: `DEBUG`, `INFO`, `WARNING`, `ERROR`, `CRITICAL`.
- Standard-Level: `INFO`.
- CLI bevorzugt: `--log-level INFO`.
- Ohne CLI: Umgebungsvariable `LOG_LEVEL`.
- Keine Tokens, Passwoerter, Secrets oder privaten Nutzdaten loggen.
- Logdateien gehoeren in `logs/` und werden nicht versioniert.

Minimalmuster:

```python
import argparse
import logging
import os


def setup_logging(level: str = "INFO") -> None:
    """Initialisiert konsistentes Logging fuer Konsole und Debugging."""
    numeric_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log-level",
        default=os.getenv("LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Programm gestartet")


if __name__ == "__main__":
    main()
```

Loggen:

- `DEBUG`: technische Details, Zwischenschritte, Pfade, Parameter ohne Secrets.
- `INFO`: Start, Ende, wichtige Arbeitsschritte, Ergebnis.
- `WARNING`: unerwartet, aber behandelbar.
- `ERROR`: fehlgeschlagene Aktion mit Kontext.
- `CRITICAL`: Programm kann nicht sinnvoll weiterlaufen.

## README.md

Kurz und praktisch.

Die README muss das Ziel der App eindeutig beschreiben. Agentische KIs muessen daraus erkennen, worauf jede Aenderung hinarbeiten soll.

Mindestinhalt:

- Ziel der App: konkreter Nutzen, Zielnutzer, Hauptworkflow, erwartetes Ergebnis
- Installation
- Start
- Konfiguration
- Beispiele
- Projektstruktur
- Pruefung/Test

Zielbeschreibung:

- Ein bis drei klare Saetze.
- Konkret statt allgemein.
- Beschreiben, was die App am Ende fuer Nutzer leisten soll.
- Grenzen nennen, wenn bestimmte Dinge bewusst nicht Ziel sind.
- Keine langen Architekturtexte. Nur dokumentieren, was beim Nutzen und Aendern hilft.

Beispiel:

```text
Ziel: Diese App wandelt lokale Audiodateien in gut lesbaren deutschen Text um.
Sie richtet sich an Nutzer, die Interviews, Notizen oder Meetings offline transkribieren wollen.
Alle Aenderungen sollen die lokale Transkription einfacher, robuster oder schneller machen.
```

## setup_venv.ps1

Soll moeglichst:

- Python-Version pruefen.
- `.venv/` erstellen.
- `pip` aktualisieren.
- `requirements.txt` installieren.
- benoetigte lokale Ordner erstellen, z. B. `input/`, `output/`, `logs/`.
- Aktivierung und Startbefehl anzeigen.

Neue Projekte nutzen `.venv/`. Bestehende Projekte mit `venv/` nicht ohne Grund umstellen.

## requirements.txt

- Nur benoetigte Laufzeitabhängigkeiten eintragen.
- Kommentare fuer besondere Installationshinweise nutzen.
- Optionale Abhaengigkeiten klar markieren.
- Keine unbenutzten Pakete behalten.
- Versionen nur pinnen, wenn Stabilitaet oder Kompatibilitaet es verlangt.

## .gitignore

Minimum:

```gitignore
.venv/
venv/
__pycache__/
*.pyc
.pytest_cache/
.mypy_cache/
.ruff_cache/
dist/
build/
*.egg-info/
logs/
*.log
output/
outputs/
input/
.env
.env.*
```

Projektabhängig ergänzen: Datenbanken, Modell-Caches, temporaere Dateien, Medien, Exporte, PDFs, private Daten.

## Konfiguration

- Defaults im Code halten, solange sie uebersichtlich sind.
- Lokale Overrides ueber `.env` oder nicht versionierte JSON/YAML.
- Beispiele als `.env.example` oder `config.example.*`.
- Keine Secrets committen.
- Zugangsdaten nie hart codieren.

## Tests und Pruefung

Pragmatisch pruefen. Keine Teststruktur erzwingen, wenn ein klarer Startbefehl reicht.

Minimum im README:

```powershell
.\setup_venv.ps1
.\.venv\Scripts\Activate.ps1
python main.py --help
python main.py
```

Wenn Tests vorhanden sind:

```powershell
python -m pytest
```

Nur Befehle dokumentieren, die wirklich funktionieren.

## Wann auslagern?

Auslagern nur wenn mindestens einer dieser Punkte zutrifft:

- Eine Datei wird schwer zu lesen.
- Eine Funktion wird mehrfach gebraucht.
- Ein klar abgrenzbarer Teil kann allein getestet werden.
- Externe Schnittstellen wuerden den Hauptablauf verdecken.

Dann einfach auslagern:

```text
projekt/
├── main.py
├── helpers.py
├── README.md
├── requirements.txt
├── setup_venv.ps1
└── .gitignore
```

Keine tiefen Ordnerbaeume. Keine kuenstlichen Schichten.

## Abschlusscheck

- Projekt startet.
- Einstiegspunkt ist klar.
- Code ist lesbar und knapp kommentiert.
- Logging ist vorhanden und Log-Level ist steuerbar.
- Aktualitaet und Performance wurden geprueft und naheliegend verbessert.
- README enthaelt klares App-Ziel, Setup, Start und Pruefung.
- `.gitignore` schuetzt lokale Artefakte.
- Keine Secrets oder generierten Massendaten hinzugefuegt.
- Keine unnoetige Architektur eingefuehrt.

## Agent-Anweisung

Bei Arbeiten in diesem Workspace:

1. Kontextdateien lesen.
2. Vorhandene Muster respektieren.
3. Monolithisch und uebersichtlich arbeiten.
4. Funktion vor Architektur.
5. Kommentare und Docstrings gezielt setzen.
6. Logging mit Log-Level einbauen, wenn Laufzeitlogik geaendert wird.
7. Aktualitaet und Performance bei jeder Aenderung pruefen und naheliegend verbessern.
8. Setup, Start und Pruefung aktualisieren, wenn betroffen.
9. Am Ende knapp nennen: geaenderte Datei(en), Pruefung, offene Punkte.
