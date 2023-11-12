# Natural Language Toolkit: Language Codes
#
# Copyright (C) 2022-2023 NLTK Project
# Author: Eric Kafe <kafe.eric@gmail.com>
# URL: <https://www.nltk.org/>
# For license information, see LICENSE.TXT
#
# iso639-3 language codes (C) https://iso639-3.sil.org/

"""
Translate between language names and language codes.

The iso639-3 language codes were downloaded from the registration authority at
https://iso639-3.sil.org/

The iso639-3 codeset is evolving, so retired language codes are kept in the
"iso639retired" dictionary, which is used as fallback by the wrapper functions
"langname" and "langcode", in order to support the lookup of retired codes.

The "langcode" function returns the current iso639-3 code if there is one,
and falls back to the retired code otherwise. As specified by BCP-47,
it returns the shortest (2-letter) code by default, but 3-letter codes
are also available:

    >>> import nltk.langnames as lgn
    >>> lgn.langname('fri')          #'fri' is a retired code
    'Western Frisian'

    The current code is different from the retired one:
    >>> lgn.langcode('Western Frisian')
    'fy'

    >>> lgn.langcode('Western Frisian', typ = 3)
    'fry'

"""

import re
from warnings import warn

from nltk.corpus import bcp47

codepattern = re.compile("[a-z][a-z][a-z]?")


def langname(tag, typ="full"):
    """
    Convert a composite BCP-47 tag to a language name

    >>> from nltk.langnames import langname
    >>> langname('ca-Latn-ES-valencia')
    'Catalan: Latin: Spain: Valencian'

    >>> langname('ca-Latn-ES-valencia', typ="short")
    'Catalan'
    """
    tags = tag.split("-")
    code = tags[0].lower()
    if codepattern.fullmatch(code):
        if code in iso639retired:  # retired codes
            return iso639retired[code]
        elif code in iso639short:  # 3-letter codes
            code2 = iso639short[code]  # convert to 2-letter code
            warn(f"Shortening {code!r} to {code2!r}", stacklevel=2)
            tag = "-".join([code2] + tags[1:])
        name = bcp47.name(tag)  # parse according to BCP-47
        if typ == "full":
            return name  # include all subtags
        elif name:
            return name.split(":")[0]  # only the language subtag
    else:
        warn(f"Could not find code in {code!r}", stacklevel=2)


def langcode(name, typ=2):
    """
    Convert language name to iso639-3 language code. Returns the short 2-letter
    code by default, if one is available, and the 3-letter code otherwise:

    >>> from nltk.langnames import langcode
    >>> langcode('Modern Greek (1453-)')
    'el'

    Specify 'typ=3' to get the 3-letter code:

    >>> langcode('Modern Greek (1453-)', typ=3)
    'ell'
    """
    if name in bcp47.langcode:
        code = bcp47.langcode[name]
        if typ == 3 and code in iso639long:
            code = iso639long[code]  # convert to 3-letter code
        return code
    elif name in iso639code_retired:
        return iso639code_retired[name]
    else:
        warn(f"Could not find language in {name!r}", stacklevel=2)


# =======================================================================
# Translate betwwen Wikidata Q-codes and BCP-47 codes or names
# .......................................................................


def tag2q(tag):
    """
    Convert BCP-47 tag to Wikidata Q-code

    >>> tag2q('nds-u-sd-demv')
    'Q4289225'
    """
    return bcp47.wiki_q[tag]


def q2tag(qcode):
    """
    Convert Wikidata Q-code to BCP-47 tag

    >>> q2tag('Q4289225')
    'nds-u-sd-demv'
    """
    return wiki_bcp47[qcode]


def q2name(qcode, typ="full"):
    """
    Convert Wikidata Q-code to BCP-47 (full or short) language name

    >>> q2name('Q4289225')
    'Low German: Mecklenburg-Vorpommern'

    >>> q2name('Q4289225', "short")
    'Low German'
    """
    return langname(q2tag(qcode), typ)


def lang2q(name):
    """
    Convert simple language name to Wikidata Q-code

    >>> lang2q('Low German')
    'Q25433'
    """
    return tag2q(langcode(name))


# ======================================================================
# Data dictionaries
# ......................................................................


def inverse_dict(dic):
    """Return inverse mapping, but only if it is bijective"""
    if len(dic.keys()) == len(set(dic.values())):
        return {val: key for (key, val) in dic.items()}
    else:
        warn("This dictionary has no bijective inverse mapping.")


bcp47.load_wiki_q()  # Wikidata conversion table needs to be loaded explicitly
wiki_bcp47 = inverse_dict(bcp47.wiki_q)

iso639short = {
    "aar": "aa",
    "abk": "ab",
    "afr": "af",
    "aka": "ak",
    "amh": "am",
    "ara": "ar",
    "arg": "an",
    "asm": "as",
    "ava": "av",
    "ave": "ae",
    "aym": "ay",
    "aze": "az",
    "bak": "ba",
    "bam": "bm",
    "bel": "be",
    "ben": "bn",
    "bis": "bi",
    "bod": "bo",
    "bos": "bs",
    "bre": "br",
    "bul": "bg",
    "cat": "ca",
    "ces": "cs",
    "cha": "ch",
    "che": "ce",
    "chu": "cu",
    "chv": "cv",
    "cor": "kw",
    "cos": "co",
    "cre": "cr",
    "cym": "cy",
    "dan": "da",
    "deu": "de",
    "div": "dv",
    "dzo": "dz",
    "ell": "el",
    "eng": "en",
    "epo": "eo",
    "est": "et",
    "eus": "eu",
    "ewe": "ee",
    "fao": "fo",
    "fas": "fa",
    "fij": "fj",
    "fin": "fi",
    "fra": "fr",
    "fry": "fy",
    "ful": "ff",
    "gla": "gd",
    "gle": "ga",
    "glg": "gl",
    "glv": "gv",
    "grn": "gn",
    "guj": "gu",
    "hat": "ht",
    "hau": "ha",
    "hbs": "sh",
    "heb": "he",
    "her": "hz",
    "hin": "hi",
    "hmo": "ho",
    "hrv": "hr",
    "hun": "hu",
    "hye": "hy",
    "ibo": "ig",
    "ido": "io",
    "iii": "ii",
    "iku": "iu",
    "ile": "ie",
    "ina": "ia",
    "ind": "id",
    "ipk": "ik",
    "isl": "is",
    "ita": "it",
    "jav": "jv",
    "jpn": "ja",
    "kal": "kl",
    "kan": "kn",
    "kas": "ks",
    "kat": "ka",
    "kau": "kr",
    "kaz": "kk",
    "khm": "km",
    "kik": "ki",
    "kin": "rw",
    "kir": "ky",
    "kom": "kv",
    "kon": "kg",
    "kor": "ko",
    "kua": "kj",
    "kur": "ku",
    "lao": "lo",
    "lat": "la",
    "lav": "lv",
    "lim": "li",
    "lin": "ln",
    "lit": "lt",
    "ltz": "lb",
    "lub": "lu",
    "lug": "lg",
    "mah": "mh",
    "mal": "ml",
    "mar": "mr",
    "mkd": "mk",
    "mlg": "mg",
    "mlt": "mt",
    "mon": "mn",
    "mri": "mi",
    "msa": "ms",
    "mya": "my",
    "nau": "na",
    "nav": "nv",
    "nbl": "nr",
    "nde": "nd",
    "ndo": "ng",
    "nep": "ne",
    "nld": "nl",
    "nno": "nn",
    "nob": "nb",
    "nor": "no",
    "nya": "ny",
    "oci": "oc",
    "oji": "oj",
    "ori": "or",
    "orm": "om",
    "oss": "os",
    "pan": "pa",
    "pli": "pi",
    "pol": "pl",
    "por": "pt",
    "pus": "ps",
    "que": "qu",
    "roh": "rm",
    "ron": "ro",
    "run": "rn",
    "rus": "ru",
    "sag": "sg",
    "san": "sa",
    "sin": "si",
    "slk": "sk",
    "slv": "sl",
    "sme": "se",
    "smo": "sm",
    "sna": "sn",
    "snd": "sd",
    "som": "so",
    "sot": "st",
    "spa": "es",
    "sqi": "sq",
    "srd": "sc",
    "srp": "sr",
    "ssw": "ss",
    "sun": "su",
    "swa": "sw",
    "swe": "sv",
    "tah": "ty",
    "tam": "ta",
    "tat": "tt",
    "tel": "te",
    "tgk": "tg",
    "tgl": "tl",
    "tha": "th",
    "tir": "ti",
    "ton": "to",
    "tsn": "tn",
    "tso": "ts",
    "tuk": "tk",
    "tur": "tr",
    "twi": "tw",
    "uig": "ug",
    "ukr": "uk",
    "urd": "ur",
    "uzb": "uz",
    "ven": "ve",
    "vie": "vi",
    "vol": "vo",
    "wln": "wa",
    "wol": "wo",
    "xho": "xh",
    "yid": "yi",
    "yor": "yo",
    "zha": "za",
    "zho": "zh",
    "zul": "zu",
}


iso639retired = {
    "fri": "Western Frisian",
    "auv": "Auvergnat",
    "gsc": "Gascon",
    "lms": "Limousin",
    "lnc": "Languedocien",
    "prv": "Provençal",
    "amd": "Amapá Creole",
    "bgh": "Bogan",
    "bnh": "Banawá",
    "bvs": "Belgian Sign Language",
    "ccy": "Southern Zhuang",
    "cit": "Chittagonian",
    "flm": "Falam Chin",
    "jap": "Jaruára",
    "kob": "Kohoroxitari",
    "mob": "Moinba",
    "mzf": "Aiku",
    "nhj": "Tlalitzlipa Nahuatl",
    "nhs": "Southeastern Puebla Nahuatl",
    "occ": "Occidental",
    "tmx": "Tomyang",
    "tot": "Patla-Chicontla Totonac",
    "xmi": "Miarrã",
    "yib": "Yinglish",
    "ztc": "Lachirioag Zapotec",
    "atf": "Atuence",
    "bqe": "Navarro-Labourdin Basque",
    "bsz": "Souletin Basque",
    "aex": "Amerax",
    "ahe": "Ahe",
    "aiz": "Aari",
    "akn": "Amikoana",
    "arf": "Arafundi",
    "azr": "Adzera",
    "bcx": "Pamona",
    "bii": "Bisu",
    "bke": "Bengkulu",
    "blu": "Hmong Njua",
    "boc": "Bakung Kenyah",
    "bsd": "Sarawak Bisaya",
    "bwv": "Bahau River Kenyah",
    "bxt": "Buxinhua",
    "byu": "Buyang",
    "ccx": "Northern Zhuang",
    "cru": "Carútana",
    "dat": "Darang Deng",
    "dyk": "Land Dayak",
    "eni": "Enim",
    "fiz": "Izere",
    "gen": "Geman Deng",
    "ggh": "Garreh-Ajuran",
    "itu": "Itutang",
    "kds": "Lahu Shi",
    "knh": "Kayan River Kenyah",
    "krg": "North Korowai",
    "krq": "Krui",
    "kxg": "Katingan",
    "lmt": "Lematang",
    "lnt": "Lintang",
    "lod": "Berawan",
    "mbg": "Northern Nambikuára",
    "mdo": "Southwest Gbaya",
    "mhv": "Arakanese",
    "miv": "Mimi",
    "mqd": "Madang",
    "nky": "Khiamniungan Naga",
    "nxj": "Nyadu",
    "ogn": "Ogan",
    "ork": "Orokaiva",
    "paj": "Ipeka-Tapuia",
    "pec": "Southern Pesisir",
    "pen": "Penesak",
    "plm": "Palembang",
    "poj": "Lower Pokomo",
    "pun": "Pubian",
    "rae": "Ranau",
    "rjb": "Rajbanshi",
    "rws": "Rawas",
    "sdd": "Semendo",
    "sdi": "Sindang Kelingi",
    "skl": "Selako",
    "slb": "Kahumamahon Saluan",
    "srj": "Serawai",
    "suf": "Tarpia",
    "suh": "Suba",
    "suu": "Sungkai",
    "szk": "Sizaki",
    "tle": "Southern Marakwet",
    "tnj": "Tanjong",
    "ttx": "Tutong 1",
    "ubm": "Upper Baram Kenyah",
    "vky": "Kayu Agung",
    "vmo": "Muko-Muko",
    "wre": "Ware",
    "xah": "Kahayan",
    "xkm": "Mahakam Kenyah",
    "xuf": "Kunfal",
    "yio": "Dayao Yi",
    "ymj": "Muji Yi",
    "ypl": "Pula Yi",
    "ypw": "Puwa Yi",
    "ywm": "Wumeng Yi",
    "yym": "Yuanjiang-Mojiang Yi",
    "mly": "Malay (individual language)",
    "muw": "Mundari",
    "xst": "Silt'e",
    "ope": "Old Persian",
    "scc": "Serbian",
    "scr": "Croatian",
    "xsk": "Sakan",
    "mol": "Moldavian",
    "aay": "Aariya",
    "acc": "Cubulco Achí",
    "cbm": "Yepocapa Southwestern Cakchiquel",
    "chs": "Chumash",
    "ckc": "Northern Cakchiquel",
    "ckd": "South Central Cakchiquel",
    "cke": "Eastern Cakchiquel",
    "ckf": "Southern Cakchiquel",
    "cki": "Santa María De Jesús Cakchiquel",
    "ckj": "Santo Domingo Xenacoj Cakchiquel",
    "ckk": "Acatenango Southwestern Cakchiquel",
    "ckw": "Western Cakchiquel",
    "cnm": "Ixtatán Chuj",
    "cti": "Tila Chol",
    "cun": "Cunén Quiché",
    "eml": "Emiliano-Romagnolo",
    "eur": "Europanto",
    "gmo": "Gamo-Gofa-Dawro",
    "hsf": "Southeastern Huastec",
    "hva": "San Luís Potosí Huastec",
    "ixi": "Nebaj Ixil",
    "ixj": "Chajul Ixil",
    "jai": "Western Jacalteco",
    "mms": "Southern Mam",
    "mpf": "Tajumulco Mam",
    "mtz": "Tacanec",
    "mvc": "Central Mam",
    "mvj": "Todos Santos Cuchumatán Mam",
    "poa": "Eastern Pokomam",
    "pob": "Western Pokomchí",
    "pou": "Southern Pokomam",
    "ppv": "Papavô",
    "quj": "Joyabaj Quiché",
    "qut": "West Central Quiché",
    "quu": "Eastern Quiché",
    "qxi": "San Andrés Quiché",
    "sic": "Malinguat",
    "stc": "Santa Cruz",
    "tlz": "Toala'",
    "tzb": "Bachajón Tzeltal",
    "tzc": "Chamula Tzotzil",
    "tze": "Chenalhó Tzotzil",
    "tzs": "San Andrés Larrainzar Tzotzil",
    "tzt": "Western Tzutujil",
    "tzu": "Huixtán Tzotzil",
    "tzz": "Zinacantán Tzotzil",
    "vlr": "Vatrata",
    "yus": "Chan Santa Cruz Maya",
    "nfg": "Nyeng",
    "nfk": "Shakara",
    "agp": "Paranan",
    "bhk": "Albay Bicolano",
    "bkb": "Finallig",
    "btb": "Beti (Cameroon)",
    "cjr": "Chorotega",
    "cmk": "Chimakum",
    "drh": "Darkhat",
    "drw": "Darwazi",
    "gav": "Gabutamon",
    "mof": "Mohegan-Montauk-Narragansett",
    "mst": "Cataelano Mandaya",
    "myt": "Sangab Mandaya",
    "rmr": "Caló",
    "sgl": "Sanglechi-Ishkashimi",
    "sul": "Surigaonon",
    "sum": "Sumo-Mayangna",
    "tnf": "Tangshewi",
    "wgw": "Wagawaga",
    "ayx": "Ayi (China)",
    "bjq": "Southern Betsimisaraka Malagasy",
    "dha": "Dhanwar (India)",
    "dkl": "Kolum So Dogon",
    "mja": "Mahei",
    "nbf": "Naxi",
    "noo": "Nootka",
    "tie": "Tingal",
    "tkk": "Takpa",
    "baz": "Tunen",
    "bjd": "Bandjigali",
    "ccq": "Chaungtha",
    "cka": "Khumi Awa Chin",
    "dap": "Nisi (India)",
    "dwl": "Walo Kumbe Dogon",
    "elp": "Elpaputih",
    "gbc": "Garawa",
    "gio": "Gelao",
    "hrr": "Horuru",
    "ibi": "Ibilo",
    "jar": "Jarawa (Nigeria)",
    "kdv": "Kado",
    "kgh": "Upper Tanudan Kalinga",
    "kpp": "Paku Karen",
    "kzh": "Kenuzi-Dongola",
    "lcq": "Luhu",
    "mgx": "Omati",
    "nln": "Durango Nahuatl",
    "pbz": "Palu",
    "pgy": "Pongyong",
    "sca": "Sansu",
    "tlw": "South Wemale",
    "unp": "Worora",
    "wiw": "Wirangu",
    "ybd": "Yangbye",
    "yen": "Yendang",
    "yma": "Yamphe",
    "daf": "Dan",
    "djl": "Djiwarli",
    "ggr": "Aghu Tharnggalu",
    "ilw": "Talur",
    "izi": "Izi-Ezaa-Ikwo-Mgbo",
    "meg": "Mea",
    "mld": "Malakhel",
    "mnt": "Maykulan",
    "mwd": "Mudbura",
    "myq": "Forest Maninka",
    "nbx": "Ngura",
    "nlr": "Ngarla",
    "pcr": "Panang",
    "ppr": "Piru",
    "tgg": "Tangga",
    "wit": "Wintu",
    "xia": "Xiandao",
    "yiy": "Yir Yoront",
    "yos": "Yos",
    "emo": "Emok",
    "ggm": "Gugu Mini",
    "leg": "Lengua",
    "lmm": "Lamam",
    "mhh": "Maskoy Pidgin",
    "puz": "Purum Naga",
    "sap": "Sanapaná",
    "yuu": "Yugh",
    "aam": "Aramanik",
    "adp": "Adap",
    "aue": "ǂKxʼauǁʼein",
    "bmy": "Bemba (Democratic Republic of Congo)",
    "bxx": "Borna (Democratic Republic of Congo)",
    "byy": "Buya",
    "dzd": "Daza",
    "gfx": "Mangetti Dune ǃXung",
    "gti": "Gbati-ri",
    "ime": "Imeraguen",
    "kbf": "Kakauhua",
    "koj": "Sara Dunjo",
    "kwq": "Kwak",
    "kxe": "Kakihum",
    "lii": "Lingkhim",
    "mwj": "Maligo",
    "nnx": "Ngong",
    "oun": "ǃOǃung",
    "pmu": "Mirpur Panjabi",
    "sgo": "Songa",
    "thx": "The",
    "tsf": "Southwestern Tamang",
    "uok": "Uokha",
    "xsj": "Subi",
    "yds": "Yiddish Sign Language",
    "ymt": "Mator-Taygi-Karagas",
    "ynh": "Yangho",
    "bgm": "Baga Mboteni",
    "btl": "Bhatola",
    "cbe": "Chipiajes",
    "cbh": "Cagua",
    "coy": "Coyaima",
    "cqu": "Chilean Quechua",
    "cum": "Cumeral",
    "duj": "Dhuwal",
    "ggn": "Eastern Gurung",
    "ggo": "Southern Gondi",
    "guv": "Gey",
    "iap": "Iapama",
    "ill": "Iranun",
    "kgc": "Kasseng",
    "kox": "Coxima",
    "ktr": "Kota Marudu Tinagas",
    "kvs": "Kunggara",
    "kzj": "Coastal Kadazan",
    "kzt": "Tambunan Dusun",
    "nad": "Nijadali",
    "nts": "Natagaimas",
    "ome": "Omejes",
    "pmc": "Palumata",
    "pod": "Ponares",
    "ppa": "Pao",
    "pry": "Pray 3",
    "rna": "Runa",
    "svr": "Savara",
    "tdu": "Tempasuk Dusun",
    "thc": "Tai Hang Tong",
    "tid": "Tidong",
    "tmp": "Tai Mène",
    "tne": "Tinoc Kallahan",
    "toe": "Tomedes",
    "xba": "Kamba (Brazil)",
    "xbx": "Kabixí",
    "xip": "Xipináwa",
    "xkh": "Karahawyana",
    "yri": "Yarí",
    "jeg": "Jeng",
    "kgd": "Kataang",
    "krm": "Krim",
    "prb": "Lua'",
    "puk": "Pu Ko",
    "rie": "Rien",
    "rsi": "Rennellese Sign Language",
    "skk": "Sok",
    "snh": "Shinabo",
    "lsg": "Lyons Sign Language",
    "mwx": "Mediak",
    "mwy": "Mosiro",
    "ncp": "Ndaktup",
    "ais": "Nataoran Amis",
    "asd": "Asas",
    "dit": "Dirari",
    "dud": "Hun-Saare",
    "lba": "Lui",
    "llo": "Khlor",
    "myd": "Maramba",
    "myi": "Mina (India)",
    "nns": "Ningye",
    "aoh": "Arma",
    "ayy": "Tayabas Ayta",
    "bbz": "Babalia Creole Arabic",
    "bpb": "Barbacoas",
    "cca": "Cauca",
    "cdg": "Chamari",
    "dgu": "Degaru",
    "drr": "Dororo",
    "ekc": "Eastern Karnic",
    "gli": "Guliguli",
    "kjf": "Khalaj",
    "kxl": "Nepali Kurux",
    "kxu": "Kui (India)",
    "lmz": "Lumbee",
    "nxu": "Narau",
    "plp": "Palpa",
    "sdm": "Semandang",
    "tbb": "Tapeba",
    "xrq": "Karranga",
    "xtz": "Tasmanian",
    "zir": "Ziriya",
    "thw": "Thudam",
    "bic": "Bikaru",
    "bij": "Vaghat-Ya-Bijim-Legeri",
    "blg": "Balau",
    "gji": "Geji",
    "mvm": "Muya",
    "ngo": "Ngoni",
    "pat": "Papitalai",
    "vki": "Ija-Zuba",
    "wra": "Warapu",
    "ajt": "Judeo-Tunisian Arabic",
    "cug": "Chungmboko",
    "lak": "Laka (Nigeria)",
    "lno": "Lango (South Sudan)",
    "pii": "Pini",
    "smd": "Sama",
    "snb": "Sebuyau",
    "uun": "Kulon-Pazeh",
    "wrd": "Warduji",
    "wya": "Wyandot",
}


iso639long = inverse_dict(iso639short)

iso639code_retired = inverse_dict(iso639retired)
