"""Per-language locale pools and name pools.

WHY: without an explicit place/context, a generation model defaults to a
generic (often Western) setting -- "a city", "a supermarket", "John". Each
sampled example therefore gets a concrete, culturally-grounded locale drawn
from where the language is actually spoken, plus a few common personal names
to use for fictional characters, so that names/places/currency/units in the
generated text are local rather than imported.

Locale labels are unique *within* a language (they are used as values of the
`locale` sampling attribute). Countries carry the currency information that
goes into the prompt. All content here is hand-written general knowledge;
extend freely -- the sampler picks up new entries automatically.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Country:
    code: str
    name: str
    currency: str  # how money should be written in text
    notes: str  # short cultural/practical cues for the prompt


COUNTRIES: dict[str, Country] = {
    "NG": Country("NG", "Nigeria", "naira (₦), with kobo for small change", "distances in kilometres; power cuts, generators, keke/okada/danfo transport are everyday realities"),
    "GH": Country("GH", "Ghana", "Ghana cedi (GH₵), with pesewas", "trotro minibuses, chop bars, mobile money (MoMo) and market days are everyday realities"),
    "BJ": Country("BJ", "Benin Republic", "CFA franc (FCFA, XOF)", "zémidjan motorbike taxis, large open-air markets such as Dantokpa; French is the official language"),
    "TG": Country("TG", "Togo", "CFA franc (FCFA, XOF)", "zémidjan motorbike taxis, market women and French as the official language"),
    "CM": Country("CM", "Cameroon", "CFA franc (FCFA, XAF)", "cattle trade, weekly markets, French and English as official languages"),
    "SN": Country("SN", "Senegal", "CFA franc (FCFA, XOF)", "French as the official language, Wolof as the main lingua franca, Sufi brotherhoods"),
    "GN": Country("GN", "Guinea", "Guinean franc (GNF)", "the Fouta Djallon highlands, French as the official language"),
    "ML": Country("ML", "Mali", "CFA franc (FCFA, XOF)", "Niger River trade, cattle and millet farming, French as the official language"),
    "MR": Country("MR", "Mauritania", "ouguiya (MRU)", "pastoralism along the Senegal River valley, Arabic and French"),
    "GM": Country("GM", "The Gambia", "dalasi (GMD)", "riverine farming and trading, English as the official language"),
    "NE": Country("NE", "Niger", "CFA franc (FCFA, XOF)", "Sahelian farming and pastoralism, French as the official language"),
}


@dataclass(frozen=True)
class Locale:
    label: str  # unique within a language; shown to the model
    country: str  # key into COUNTRIES


def _l(label: str, country: str) -> Locale:
    return Locale(label, country)


LOCALES: dict[str, list[Locale]] = {
    "yor": [
        _l("Lagos Island and the Lagos mainland (Yaba, Surulere, Ikeja)", "NG"),
        _l("Ikorodu, Lagos State", "NG"),
        _l("Badagry, Lagos State", "NG"),
        _l("Ibadan, Oyo State", "NG"),
        _l("Ogbomoso, Oyo State", "NG"),
        _l("Oyo town, Oyo State", "NG"),
        _l("Abeokuta, Ogun State", "NG"),
        _l("Ijebu-Ode, Ogun State", "NG"),
        _l("a farming village in Ogun State", "NG"),
        _l("Osogbo, Osun State", "NG"),
        _l("Ile-Ife, Osun State", "NG"),
        _l("Ilesa, Osun State", "NG"),
        _l("Akure, Ondo State", "NG"),
        _l("Ondo town, Ondo State", "NG"),
        _l("Ado-Ekiti, Ekiti State", "NG"),
        _l("Ilorin, Kwara State (Yoruba community)", "NG"),
        _l("Porto-Novo and Ketu, Benin Republic (Yoruba communities)", "BJ"),
        _l("Sakété, Benin Republic", "BJ"),
        _l("a small Yoruba town in Oyo State on market day", "NG"),
        _l("a university campus in southwest Nigeria", "NG"),
    ],
    "hau": [
        _l("Kano city (Sabon Gari, Kurmi Market, Fagge)", "NG"),
        _l("Kaduna city", "NG"),
        _l("Zaria, Kaduna State", "NG"),
        _l("Sokoto city", "NG"),
        _l("Katsina city", "NG"),
        _l("Gusau, Zamfara State", "NG"),
        _l("Birnin Kebbi, Kebbi State", "NG"),
        _l("Dutse, Jigawa State", "NG"),
        _l("Bauchi city", "NG"),
        _l("Gombe city", "NG"),
        _l("Maiduguri, Borno State", "NG"),
        _l("Minna, Niger State", "NG"),
        _l("Jos, Plateau State (Hausa community)", "NG"),
        _l("Abuja (Hausa community in Wuse, Kubwa)", "NG"),
        _l("a farming village in northern Nigeria during the rainy season", "NG"),
        _l("Maradi, Niger Republic", "NE"),
        _l("Zinder, Niger Republic", "NE"),
        _l("Niamey, Niger Republic", "NE"),
    ],
    "ibo": [
        _l("Enugu city", "NG"),
        _l("Nsukka, Enugu State", "NG"),
        _l("Onitsha, Anambra State (Main Market)", "NG"),
        _l("Nnewi, Anambra State", "NG"),
        _l("Awka, Anambra State", "NG"),
        _l("Owerri, Imo State", "NG"),
        _l("Orlu, Imo State", "NG"),
        _l("Aba, Abia State (Ariaria Market)", "NG"),
        _l("Umuahia, Abia State", "NG"),
        _l("Arochukwu, Abia State", "NG"),
        _l("Abakaliki, Ebonyi State", "NG"),
        _l("Asaba, Delta State (Igbo community)", "NG"),
        _l("Port Harcourt (Igbo community)", "NG"),
        _l("Alaba and Ojo markets, Lagos (Igbo traders)", "NG"),
        _l("an Igbo village during the New Yam festival season", "NG"),
        _l("a village during the end-of-year town-union meeting when people return from the cities", "NG"),
    ],
    "efi": [
        _l("Calabar South and Calabar Municipality, Cross River State", "NG"),
        _l("Duke Town and Creek Town (Old Calabar)", "NG"),
        _l("Akpabuyo, Cross River State", "NG"),
        _l("Odukpani, Cross River State", "NG"),
        _l("Ikom, Cross River State", "NG"),
        _l("Ogoja, Cross River State", "NG"),
        _l("Obudu, Cross River State", "NG"),
        _l("Uyo, Akwa Ibom State", "NG"),
        _l("Eket, Akwa Ibom State", "NG"),
        _l("Oron, Akwa Ibom State", "NG"),
        _l("a riverside fishing community near Calabar", "NG"),
        _l("the University of Calabar campus", "NG"),
        _l("Marian Market and Watt Market, Calabar", "NG"),
        _l("Calabar during the December carnival season", "NG"),
    ],
    "urh": [
        _l("Warri, Delta State", "NG"),
        _l("Effurun, Uvwie, Delta State", "NG"),
        _l("Sapele, Delta State", "NG"),
        _l("Ughelli, Delta State", "NG"),
        _l("Orerokpe, Okpe, Delta State", "NG"),
        _l("Abraka, Delta State", "NG"),
        _l("Agbarho, Delta State", "NG"),
        _l("Ovwian and Aladja, Udu, Delta State", "NG"),
        _l("Oghara, Delta State", "NG"),
        _l("Otor-Udu and Ekpan, Delta State", "NG"),
        _l("a riverine Urhobo community in the Niger Delta", "NG"),
        _l("Benin City (Urhobo community)", "NG"),
        _l("an Urhobo village during a festival or funeral celebration", "NG"),
        _l("Delta State University campus, Abraka", "NG"),
    ],
    "twi": [
        _l("Kumasi (Kejetia Market, Adum)", "GH"),
        _l("Ejisu and Bekwai, Ashanti Region", "GH"),
        _l("Obuasi, Ashanti Region", "GH"),
        _l("Konongo and Mampong, Ashanti Region", "GH"),
        _l("Accra (Madina, Kaneshie, Makola Market)", "GH"),
        _l("Tema, Greater Accra", "GH"),
        _l("Aburi and Akropong, Akuapem", "GH"),
        _l("Nsawam and Suhum, Eastern Region", "GH"),
        _l("Koforidua, Eastern Region", "GH"),
        _l("Nkawkaw and Mpraeso, Kwahu", "GH"),
        _l("Sunyani, Bono Region", "GH"),
        _l("Techiman, Bono East Region", "GH"),
        _l("Takoradi, Western Region", "GH"),
        _l("a cocoa-farming village in Ashanti or Eastern Region", "GH"),
        _l("KNUST campus, Kumasi", "GH"),
    ],
    "aka": [
        _l("Accra and its Akan-speaking neighbourhoods", "GH"),
        _l("Kumasi and the Ashanti Region", "GH"),
        _l("Akyem Oda and Kade, Eastern Region", "GH"),
        _l("Kwahu towns (Nkawkaw, Abetifi)", "GH"),
        _l("Dunkwa-on-Offin and Tarkwa, Western Region", "GH"),
        _l("Cape Coast and Elmina (Fante coast)", "GH"),
        _l("Winneba and Mankessim, Central Region", "GH"),
        _l("Sekondi-Takoradi", "GH"),
        _l("Wenchi and Techiman, Bono", "GH"),
        _l("Akropong, Akuapem", "GH"),
        _l("a chief's palace durbar in an Akan town", "GH"),
        _l("a fishing community on the Ghanaian coast", "GH"),
        _l("a farming village in the forest belt of Ghana", "GH"),
    ],
    "ewe": [
        _l("Ho, Volta Region", "GH"),
        _l("Hohoe, Volta Region", "GH"),
        _l("Keta and Anloga, Volta Region", "GH"),
        _l("Aflao and Denu (Ghana-Togo border)", "GH"),
        _l("Kpando, Volta Region", "GH"),
        _l("Sogakope and Adidome, Volta Region", "GH"),
        _l("Accra (Ewe community, Nima and Tudu)", "GH"),
        _l("Lomé, Togo (Grand Marché)", "TG"),
        _l("Aného, Togo", "TG"),
        _l("Kpalimé, Togo", "TG"),
        _l("Atakpamé, Togo", "TG"),
        _l("Tsévié, Togo", "TG"),
        _l("Vogan and Tabligbo, Togo", "TG"),
        _l("a fishing or farming village in the Volta Region", "GH"),
        _l("the Hogbetsotso festival season in Anlo land", "GH"),
    ],
    "fon": [
        _l("Cotonou (Dantokpa Market)", "BJ"),
        _l("Porto-Novo", "BJ"),
        _l("Abomey and Bohicon", "BJ"),
        _l("Ouidah", "BJ"),
        _l("Abomey-Calavi", "BJ"),
        _l("Allada", "BJ"),
        _l("Lokossa and Grand-Popo", "BJ"),
        _l("Dassa-Zoumè", "BJ"),
        _l("Savalou", "BJ"),
        _l("Sèmè-Podji", "BJ"),
        _l("Ganvié, the lake village near Cotonou", "BJ"),
        _l("a farming village in southern Benin", "BJ"),
        _l("Aného, Togo (Fon-speaking community)", "TG"),
        _l("the Vodun festival season in Ouidah", "BJ"),
    ],
    "pcm": [
        _l("Lagos (Yaba, Oshodi, Ikeja, Ajegunle)", "NG"),
        _l("Port Harcourt, Rivers State", "NG"),
        _l("Warri and Sapele, Delta State", "NG"),
        _l("Benin City, Edo State", "NG"),
        _l("Abuja (Wuse, Garki, Nyanya)", "NG"),
        _l("Onitsha, Anambra State", "NG"),
        _l("Aba, Abia State", "NG"),
        _l("Calabar, Cross River State", "NG"),
        _l("Uyo, Akwa Ibom State", "NG"),
        _l("Enugu city", "NG"),
        _l("Ibadan, Oyo State", "NG"),
        _l("Kano (Sabon Gari)", "NG"),
        _l("Jos, Plateau State", "NG"),
        _l("a Nigerian university campus hostel", "NG"),
        _l("a Nigerian bus park or motor garage", "NG"),
        _l("Nigerian social media (Twitter/X, WhatsApp groups)", "NG"),
        _l("Nigeria broadly (no specific city)", "NG"),
    ],
    "ful": [
        _l("Dakar, Senegal", "SN"),
        _l("Saint-Louis and the Fouta Toro valley (Podor, Matam), Senegal", "SN"),
        _l("Kaolack and the groundnut basin, Senegal", "SN"),
        _l("Labé and the Fouta Djallon highlands, Guinea", "GN"),
        _l("Mamou and Pita, Guinea", "GN"),
        _l("Conakry, Guinea", "GN"),
        _l("Bamako, Mali", "ML"),
        _l("Mopti and the inland Niger delta (Macina), Mali", "ML"),
        _l("Nouakchott and the Senegal River valley, Mauritania", "MR"),
        _l("Basse and the Upper River region, The Gambia", "GM"),
        _l("a Fulani cattle-herding camp in the Sahel", "SN"),
        _l("Sokoto, Nigeria (Fulani heartland)", "NG"),
        _l("Ngaoundéré, Adamawa Region, Cameroon", "CM"),
        _l("a weekly rural market (loumo) in Senegal", "SN"),
    ],
    "fuv": [
        _l("Yola and Jimeta, Adamawa State", "NG"),
        _l("Mubi, Adamawa State", "NG"),
        _l("Numan and Lamurde, Adamawa State", "NG"),
        _l("Jalingo, Taraba State", "NG"),
        _l("the Mambilla Plateau, Taraba State (cattle herders)", "NG"),
        _l("Gombe city", "NG"),
        _l("Bauchi city", "NG"),
        _l("Sokoto city", "NG"),
        _l("Katsina city", "NG"),
        _l("Kano (Fulani community)", "NG"),
        _l("Maiduguri, Borno State", "NG"),
        _l("Abuja (Fulfulde-speaking community)", "NG"),
        _l("Ngaoundéré, Adamawa Region, Cameroon", "CM"),
        _l("Garoua and Maroua, northern Cameroon", "CM"),
        _l("a Fulani cattle camp and seasonal transhumance route", "NG"),
        _l("a weekly cattle market in northern Nigeria", "NG"),
    ],
}

# Common given names, used only to suggest local character names for
# fictional content (a few are shown per request, randomly chosen, so names
# vary across documents). These are ordinary first names, not real persons.
NAME_POOLS: dict[str, list[str]] = {
    "yor": ["Adébáyọ̀", "Tọ́pẹ́", "Ìyábọ̀", "Bàbátúndé", "Fúnmiláyọ̀", "Ọlásúnkànmí", "Kẹ́hìndé", "Sèyí", "Àbíọ́lá", "Ṣadé", "Dàpọ̀", "Yétúndé"],
    "hau": ["Aminu", "Hadiza", "Sani", "Zainab", "Musa", "Fatima", "Ibrahim", "Rabi'u", "Hauwa", "Abubakar", "Maryam", "Yusuf"],
    "ibo": ["Chinedu", "Ngozi", "Ifeanyi", "Adaeze", "Emeka", "Chidinma", "Obinna", "Nkechi", "Uche", "Ebele", "Ikenna", "Amaka"],
    "efi": ["Etim", "Affiong", "Edet", "Ekaette", "Asuquo", "Inyang", "Mfon", "Ntiense", "Bassey", "Ememobong", "Idara", "Utibe"],
    "urh": ["Ejiro", "Oghenekaro", "Ufuoma", "Efe", "Tega", "Oghenero", "Onome", "Rume", "Ovie", "Avwerosuoghene", "Oghenetega", "Erhuvwu"],
    "twi": ["Kofi", "Ama", "Kwame", "Akosua", "Yaw", "Abena", "Kwabena", "Adwoa", "Kojo", "Afia", "Nana Yaa", "Kwaku"],
    "aka": ["Kwesi", "Efua", "Kwadwo", "Yaa", "Kofi", "Esi", "Kwaku", "Araba", "Ekow", "Akua", "Nana Ama", "Kobina"],
    "ewe": ["Kofi", "Kokou", "Afi", "Ama", "Mawuli", "Sena", "Edem", "Yawo", "Dzifa", "Selorm", "Akpene", "Kossi"],
    "fon": ["Codjo", "Adjovi", "Dossou", "Sènami", "Agossou", "Gbêdo", "Kpadonou", "Hounkpatin", "Ahouansou", "Fifamè", "Sèdjro", "Mahougnon"],
    "pcm": ["Emeka", "Tunde", "Amina", "Bisi", "Chidi", "Ngozi", "Musa", "Ebi", "Ejiro", "Ibrahim", "Blessing", "Femi"],
    "ful": ["Mamadou", "Aissatou", "Oumar", "Fatoumata", "Amadou", "Hawa", "Ibrahima", "Bintou", "Alpha", "Kadiatou", "Boubacar", "Mariama"],
    "fuv": ["Hammadu", "Aminatu", "Abubakar", "Hawwa", "Umaru", "Jamilu", "Ibrahim", "Ardo", "Bello", "Fadimatu", "Sulaiman", "Adama"],
}


def locales_for(language: str) -> list[Locale]:
    return LOCALES[language]


def locale_labels(language: str) -> list[str]:
    return [loc.label for loc in LOCALES[language]]


def locale_info(language: str, label: str) -> tuple[Locale, Country]:
    for loc in LOCALES[language]:
        if loc.label == label:
            return loc, COUNTRIES[loc.country]
    raise KeyError(f"Unknown locale {label!r} for language {language!r}")
