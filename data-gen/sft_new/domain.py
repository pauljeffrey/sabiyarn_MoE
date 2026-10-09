"""Document-domain flags by string match (no reading, no model): <|is_financial|>, <|is_medical|>, <|is_legal|>,
<|is_scientific|>, <|is_technical|>, else <|is_non_technical|>.

A flag needs at least two DISTINCT keyword hits (so one stray 'bank' does not flag a document) and the winner needs a clear lead.
Matching is on lower-cased, diacritic-stripped text with whole-word / prefix boundaries. The lists mix English (which these
languages borrow heavily, e.g. 'bank', 'virus', 'internet') with core vocabulary of each language. The lists are heuristic
and deliberately small; they exist to label the coarse domain of a document, not to be exhaustive.
"""

from __future__ import annotations

import re
import unicodedata

KEYWORDS = {
    "<|is_financial|>": """money bank banks loan loans credit debit interest budget tax taxes price prices market economy economic inflation
        invest investment shares stock stocks currency dollar dollars naira cedi rand shilling shillings franc francs profit salary
        insurance trade export import revenue fiscal trader
        pesa benki mkopo bei shilingi biashara uchumi kodi imali ibhange isikweletu intengo amarandi ibhanki amaxabiso mari bhengi mutengo
        misika geld lening prys rente belasting ekonomie kuɗi kudi banki bashi farashi kasuwa haraji owo inawo gbese ojo ego onuahia ahia
        ulo-oru lacag bangi deyn qiimaha ganacsiga amafaranga inguzanyo igiciro ubucuruzi maallaqa baankii liqaa gatii xaalis bor njeg
        sika bosea ga asikafo""",
    "<|is_medical|>": """hospital clinic doctor doctors nurse patient patients disease illness malaria covid cholera vaccine vaccination virus
        symptoms treatment medicine drug drugs pregnancy fever cancer diabetes hiv aids surgery health epidemic infection
        hospitali daktari mgonjwa ugonjwa dawa chanjo afya umalaria isibhedlela udokotela isifo imitha unyango chipatala chirwere mushonga
        hospitaal dokter siekte geneesmiddel gesondheid asibiti likita magani cuta lafiya rigakafi ile-iwosan dokita arun ogun ahuike
        ulo-ogwu dibia ọrịa ogwu isbitaal dhakhtar cudur caafimaad dawo ibitaro muganga indwara ubuzima ikigo-nderabuzima
        yaa-ayyaana dhukkuba fayyaa raggaasuu opitaal docteur feebar paj sundowo ayaresabea nyarewa kuntan""",
    "<|is_legal|>": """court courts judge judges law laws lawyer lawyers constitution constitutional legal illegal arrest arrested police
        prosecutor trial verdict sentence sentenced parliament bill act statute rights contract bail jail prison crime criminal
        mahakama hakimu sheria wakili katiba isigele inkantolo umthetho ummeli umgaqo-siseko inkundla mutemo gweta dare mutongi
        hof regter wet prokureur grondwet hukuma alkali doka lauya kotu ile-ejo adajo ofin agbejoro ikpe ndi-ikpe iwu
        maxkamad garsoore sharci qareen dastuur urukiko umucamanza itegeko amategeko ubutabera mana abbaa-seeraa seera abukaatoo
        yoon boroom-yoon tribinal""",
    "<|is_scientific|>": """research researchers study studies scientist scientists scientific university experiment laboratory biology chemistry physics
        climate species genome theory data analysis evidence journal discovery astronomy ecology geology mathematics
        utafiti mwanasayansi sayansi chuo-kikuu isayensi ucwaningo umcwaningi ukhemisi unzululwazi sainzi wetenschap navorsing
        universiteit wetenskaplike kimiyya bincike masanin jami'a imo-sayensi iwadi onimo-sayensi nnyocha sayensi cilmi baaritaan
        saynisyahan jaamacad ubushakashatsi ubumenyi ikoranabuhanga barumsa qorannoo saayinsii ceeb""",
    "<|is_technical|>": """internet computer computers software hardware network networks app apps phone phones smartphone mobile data digital
        online website server database technology engineer engineering code programming algorithm electricity solar battery
        wireless satellite cyber email whatsapp facebook google android
        mtandao kompyuta simu teknolojia isoftware ikhompyutha ucingo ubuchwepheshe sellafoni rekenaar selfoon tegnologie
        waya wayar hannu fasaha kwamfuta kọmputa ero-ayelujara ero-ibanisoro fon ekwentị kọmputa kompuutar telefoon teknoolajiyad
        mudasi ikoranabuhanga telefone kompiita teekinooloojii fon-ngir internet""",
}
_STRIP = re.compile(r"[̀-ͯ]")
_FLAGS = list(KEYWORDS)
_PATS = {f: re.compile(r"(?<![^\W\d_])(?:" + "|".join(sorted({re.escape(_STRIP.sub('', unicodedata.normalize('NFD', k.lower()))) for k in v.split()}, key=len, reverse=True)) + r")(?![^\W\d_])")
         for f, v in KEYWORDS.items()}
NON_TECH = "<|is_non_technical|>"


def domain_flag(text: str, min_hits: int = 2, lead: int = 1) -> str:
    t = _STRIP.sub("", unicodedata.normalize("NFD", text.lower()))
    hits = {f: len(set(_PATS[f].findall(t))) for f in _FLAGS}
    ranked = sorted(hits.items(), key=lambda kv: -kv[1])
    (best, n1), (_, n2) = ranked[0], ranked[1]
    if n1 >= min_hits and n1 - n2 >= lead:
        return best
    return NON_TECH
