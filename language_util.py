# language_util.py

def extract_language(response):
    language_map = {
        "arabic": "Arabic",
        "azerbaijani": "Azerbaijani",
        "bengali": "Bengali",
        "bosnian": "Bosnian",
        "bulgarian": "Bulgarian",
        "catalan": "Catalan",
        "chinese": "Chinese",
        "croatian": "Croatian",
        "czech": "Czech",
        "danish": "Danish",
        "dutch": "Dutch",
        "english": "English",
        "estonian": "Estonian",
        "farsi": "Farsi",
        "filipino": "Filipino",
        "finnish": "Finnish",
        "french": "French",
        "german": "German",
        "greek": "Greek",
        "gujarati": "Gujarati",
        "hebrew": "Hebrew",
        "hindi": "Hindi",
        "hungarian": "Hungarian",
        "icelandic": "Icelandic",
        "indonesian": "Indonesian",
        "italian": "Italian",
        "japanese": "Japanese",
        "kannada": "Kannada",
        "kazakh": "Kazakh",
        "korean": "Korean",
        "latvian": "Latvian",
        "lithuanian": "Lithuanian",
        "macedonian": "Macedonian",
        "malay": "Malay",
        "marathi": "Marathi",
        "norwegian": "Norwegian",
        "polish": "Polish",
        "portuguese": "Portuguese",
        "punjabi": "Punjabi",
        "romanian": "Romanian",
        "russian": "Russian",
        "serbian": "Serbian",
        "slovak": "Slovak",
        "slovenian": "Slovenian",
        "spanish": "Spanish",
        "swahili": "Swahili",
        "swedish": "Swedish",
        "tagalog": "Tagalog",
        "tamil": "Tamil",
        "telegu": "Telegu",
        "thai": "Thai",
        "turkish": "Turkish",
        "ukrainian": "Ukrainian",
        "urdu": "Urdu",
        "uzbek": "Uzbek",
        "vietnamese": "Vietnamese"
    }
    
    language = response.strip().split('.')[0].lower()
    
    # Iterate over the language map to find a match
    for key, name in language_map.items():
        if key in language:
            return name
    
    # If no match is found, return a label indicating unknown or no artist found
    if "unknown" in language or "no information available" in language or "not found" in language:
        return "No Artist Found"
    
    # If no language match is found, return the original input capitalized
    return language.capitalize()
