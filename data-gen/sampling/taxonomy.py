"""Hand-written topic taxonomy + attribute vocabularies for the corpus kinds.

WHY hand-written: an LLM asked for "a diverse document" collapses onto the
same ten topics (weather, health tips, "the importance of education").
Diversity has to be imposed from outside, so we enumerate the space
explicitly -- domains x sub-topics as the *primary* key, and genres,
registers, audiences, length buckets, perspectives, eras, difficulty levels
and locales as *secondary* attributes -- and let `sampling/sampler.py` walk
that space evenly. Sub-topics are written in English (the prompt language);
the generation model writes the content in the target language.

Compatibility: some combinations are nonsense (a "product description" of
the causes of the Benin Empire's decline, a "sermon" about tractor
maintenance). Domains carry coarse `tags`; genres/audiences/eras/tasks
declare which tags they need or refuse. The rules are data, not code, so
they are easy to audit and extend.

Domain tag vocabulary:
  technical  dense technical/scientific content
  commerce   money, products, business, markets
  academic   suited to lessons, lectures, textbooks
  culture    heritage, arts, customs, oral tradition
  personal   home, relationships, individual life
  civic      public life, institutions, services
  risk       topics where a request could touch harmful territory
             (used to place "safe decline" SFT tasks)
  numeric    naturally involves quantities (math/statistics tasks)
  historical about the past
  timeless   not tied to current events (news makes no sense)
  faith      religion / ethics / spirituality
  rural      village, farming, pastoral life
  modern_only inherently contemporary (technology, mobile money, ...)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

# ---------------------------------------------------------------------------
# Domains and sub-topics
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Domain:
    key: str
    name: str
    group: str  # coarse grouping used for group-level weights in the yaml configs
    tags: frozenset[str]
    description: str  # one line, English, shown to the generation model
    subtopics: tuple[str, ...]


def _d(key: str, name: str, group: str, tags: str, description: str, subtopics: list[str]) -> Domain:
    return Domain(key, name, group, frozenset(tags.split()), description, tuple(subtopics))


_DOMAIN_LIST: list[Domain] = [
    # ------------------------------------------------------------------ livelihoods
    _d("agriculture_crops", "Agriculture and crop farming", "livelihoods", "academic numeric commerce rural",
       "growing food and cash crops on smallholder and commercial farms in West Africa",
       ["cassava planting and processing into garri", "yam mound farming and storing yams", "maize and millet planting seasons",
        "cocoa farming and drying beans", "rice farming in fadama or swamp land", "vegetable gardening for market",
        "using organic manure and compost", "controlling pests such as armyworm without over-spraying", "post-harvest storage to reduce grain losses",
        "groundnut and cowpea farming", "oil palm harvesting and palm oil making", "cooperative farming and buying seeds together"]),
    _d("livestock_fisheries", "Livestock, poultry and fisheries", "livelihoods", "academic numeric commerce rural",
       "rearing animals and catching or farming fish",
       ["keeping local chickens and layers", "goat and sheep rearing for festivals", "cattle herding and seasonal movement of herds",
        "fish pond farming (catfish and tilapia)", "river and lagoon fishing methods", "smoking and preserving fish",
        "feeding and vaccinating animals", "dealing with farmer-herder tensions peacefully", "piggery and rabbit keeping on a small scale",
        "bee-keeping and honey harvesting"]),
    _d("trade_markets", "Trade and markets", "livelihoods", "commerce numeric culture",
       "buying and selling in open-air markets, shops and across borders",
       ["market day routines and market women's associations", "bargaining etiquette and fair prices", "how traders keep simple records of stock",
        "cross-border trade and border markets", "wholesale versus retail buying", "hawking and street trading", "kiosk and provisions store management",
        "seasonal price changes of foodstuffs", "weights, measures and local units like the paint bucket or tuwo bowl", "trader unions and market levies"]),
    _d("entrepreneurship", "Entrepreneurship and small business", "livelihoods", "commerce numeric personal",
       "starting and growing small businesses",
       ["starting a small business with little capital", "separating business money from household money", "pricing products to make a profit",
        "keeping customers happy and coming back", "thrift and rotating savings groups (ajo, esusu, susu)", "hiring and training an apprentice",
        "selling through WhatsApp and social media", "registering a business name", "learning from a business that failed", "scaling from a stall to a shop"]),
    _d("careers_work", "Careers and work life", "livelihoods", "personal commerce academic",
       "jobs, skills and the world of work",
       ["choosing a career after secondary school", "writing a CV and preparing for an interview", "apprenticeship and learning a trade",
        "workplace etiquette and respect for seniors", "balancing a job with family duties", "skills for artisans: tailoring, welding, plumbing",
        "the National Youth Service and first jobs", "changing careers later in life", "working as a teacher, nurse or driver", "remote and freelance work"]),
    _d("crafts_artisans", "Crafts and artisans", "livelihoods", "culture commerce",
       "traditional and modern handcrafts and the people who make them",
       ["weaving cloth and strip-weaving traditions", "adire and other resist-dyeing techniques", "blacksmithing and making farm tools",
        "pottery and clay water pots", "wood carving and stools", "leatherwork and sandal making", "bead making and jewellery",
        "basket and mat weaving", "how an apprentice learns from a master craftsman", "selling crafts to tourists and online"]),
    _d("extractives_mining", "Mining, oil and gas", "livelihoods", "commerce technical numeric civic",
       "extractive industries and their communities",
       ["how crude oil is found and produced, in simple terms", "artisanal gold and tin mining", "host communities and oil companies",
        "gas flaring and local health concerns", "quarrying granite and sand", "safety rules for miners", "small-scale salt production",
        "what community benefits agreements are", "cleaning up oil spills", "jobs created by the oil and gas sector"]),
    # ------------------------------------------------------------------ health
    _d("health_medicine", "Health and medicine", "health", "academic risk civic",
       "illness, treatment and looking after the body",
       ["recognising and treating malaria", "high blood pressure and how to live with it", "diabetes: signs and daily care",
        "first aid for burns and cuts", "when to go to the clinic instead of self-medicating", "typhoid and safe food and water",
        "taking prescribed antibiotics correctly", "eye care and preventing blindness", "dental care for the family", "living with sickle cell disease",
        "what a community health worker does", "understanding a hospital visit and health insurance"]),
    _d("public_health", "Public health and prevention", "health", "civic academic numeric risk rural",
       "protecting whole communities from disease",
       ["how vaccination protects a community", "cholera prevention and handwashing", "mosquito nets and controlling mosquito breeding", "how epidemics like Ebola or Lassa fever are contained",
        "tuberculosis: symptoms and completing treatment", "HIV prevention and testing without stigma", "reducing road traffic injuries", "school health programmes and deworming",
        "food hygiene in markets and street food", "why clean air matters: cooking smoke and generators"]),
    _d("maternal_child_health", "Maternal and child health", "health", "personal civic risk academic",
       "pregnancy, birth and the health of babies and children",
       ["antenatal care and why the clinic visits matter", "breastfeeding and weaning foods", "immunisation schedule for babies", "signs of danger in pregnancy",
        "caring for a newborn at home", "treating diarrhoea with oral rehydration", "child malnutrition and locally available nutritious food", "spacing births and family planning counselling",
        "traditional birth attendants and referral to hospitals", "fever in children and when to seek care"]),
    _d("nutrition_fitness", "Nutrition, diet and fitness", "health", "personal academic",
       "eating well and staying active",
       ["a balanced plate with local foods", "beans, vegetables and affordable protein", "eating less salt and sugar", "healthy snacks instead of sugary drinks",
        "walking, farm work and exercise", "feeding children who are picky eaters", "fasting seasons and staying healthy", "reading food labels in the shop",
        "healthy weight and myths about body size", "drinking enough safe water"]),
    _d("psychology_wellbeing", "Mental wellbeing and psychology", "health", "personal academic risk",
       "feelings, stress and mental health",
       ["coping with stress and worry", "exam pressure and how to manage it", "grief and mourning", "talking about depression without shame",
        "building self-confidence", "handling anger and conflict calmly", "loneliness of moving to a new city", "sleep and a healthy routine",
        "substance abuse and how families can help", "seeking help from a counsellor or trusted elder"]),
    _d("disability_inclusion", "Disability and inclusion", "health", "civic personal risk",
       "living with disability and building inclusive communities",
       ["children with disabilities and school", "accessibility of buildings and buses", "sign language and deaf communities", "employment for people with disabilities",
        "myths and stigma about disability", "assistive devices such as wheelchairs and white canes", "blindness and braille learning", "supporting a family member with epilepsy",
        "inclusive sports and the Paralympics", "laws protecting persons with disabilities"]),
    # ------------------------------------------------------------------ society & governance
    _d("education", "Education and learning", "society", "academic civic personal",
       "schools, teachers and learning at every age",
       ["choosing a good primary school", "learning to read in the mother tongue", "study habits for exams", "the role of parents-teachers associations",
        "vocational and technical education", "adult literacy classes", "using radio and phones for learning", "school fees and scholarships",
        "teaching with limited materials", "why girls' education matters", "university admission and JAMB/WAEC or WASSCE exams", "bullying and discipline in schools"]),
    _d("law_governance", "Law and governance", "society", "civic academic risk",
       "how laws, courts and government work",
       ["what a constitution is and what rights citizens have", "how a local government council works", "the police and your rights when stopped",
        "how to settle a dispute at a customary or magistrate's court", "land documents and tenancy rights", "consumer protection and complaining properly",
        "what a will is and why it matters", "the role of legislators and how bills become law", "anti-corruption and whistleblowing", "the difference between customary and statutory law"]),
    _d("civic_life", "Civic life and community", "society", "civic culture",
       "taking part in community and national life",
       ["voting and why every vote counts", "town-union and community development associations", "paying taxes and what they fund", "volunteering for community clean-up",
        "peaceful protest and dialogue", "respect for others across ethnic and religious lines", "how to write a petition to the local council", "community policing and neighbourhood watch",
        "census and why being counted matters", "national symbols, anthems and holidays"]),
    _d("public_services_documents", "Public services and official documents", "society", "civic commerce",
       "getting things done with government offices",
       ["getting a national ID or voter's card", "applying for a passport", "registering a birth and getting a birth certificate", "driver's licence and vehicle papers",
        "how to pay utility and electricity bills", "getting a marriage certificate", "avoiding touts and fake agents at offices", "applying for a bank account with valid ID",
        "pensions and how retirees claim them", "verifying a document's authenticity"]),
    _d("security_safety", "Security and personal safety", "society", "civic risk",
       "staying safe at home, on the road and online",
       ["securing your home and neighbourhood", "safe travel on highways and at night", "recognising scams and fake alerts", "protecting your phone and SIM from theft",
        "what to do if you witness an accident", "keeping children safe from harm", "fire safety in homes and markets", "safe use of cooking gas and generators",
        "reporting crime to the authorities", "conflict prevention and mediation in communities"]),
    _d("disasters_emergencies", "Disasters and emergencies", "society", "civic risk numeric",
       "floods, fires and other emergencies and how people respond",
       ["flooding: warning signs and preparing", "market and house fires", "building collapse and safety inspection", "drought and food shortage",
        "displaced families and camps", "emergency numbers and first responders", "preparing an emergency go-bag", "community early-warning systems",
        "rebuilding after a disaster", "helping neighbours and donating responsibly"]),
    _d("gender_society", "Gender and society", "society", "civic personal risk",
       "roles, rights and relationships between women and men",
       ["women in trade and leadership", "girl-child education and early marriage", "sharing household chores fairly", "respect and consent in relationships",
        "women's cooperative savings groups", "widows' rights and inheritance", "men's roles in child care", "domestic violence: where to get help",
        "equal pay and working mothers", "changing ideas about roles across generations"]),
    _d("migration_diaspora", "Migration and diaspora", "society", "civic personal commerce risk",
       "moving within and outside West Africa",
       ["moving from the village to the city for work", "the dangers of irregular migration and human trafficking", "remittances and supporting family back home", "students studying abroad",
        "ECOWAS free movement of people and goods", "returning home after years abroad", "staying in touch by video call across countries", "learning a new language in a new place",
        "diaspora associations and hometown projects", "border crossing and travel documents"]),
    _d("traditional_authority", "Chieftaincy and traditional institutions", "society", "culture civic historical",
       "kings, chiefs, councils of elders and customary institutions",
       ["the role of a traditional ruler in a community", "how a chief is chosen and installed", "councils of elders and settling disputes", "palace etiquette and greetings",
        "regalia, stools and symbols of authority", "traditional rulers and modern government", "age-grade associations", "the role of the town crier and messenger",
        "land ownership under customary law", "secret societies and their public roles, in general terms"]),
    # ------------------------------------------------------------------ economy & money
    _d("economics_business", "Economics and business", "economy", "commerce numeric academic",
       "how the economy and companies work",
       ["inflation and why prices rise", "supply and demand in the local market", "exports of cocoa, oil and other goods", "what a budget is at household and national level",
        "importing goods and paying customs duties", "why a stronger local currency matters", "what a cooperative society is", "how a company is formed and managed",
        "the informal economy and why it is large", "jobs and unemployment among young people", "the price of fuel and its ripple effects", "fair trade and small producers"]),
    _d("finance_mobile_money", "Personal finance and mobile money", "economy", "commerce numeric risk modern_only",
       "saving, borrowing, banking and sending money",
       ["making a household budget", "how mobile money wallets work", "saving a little every day", "bank accounts, ATMs and cards", "avoiding loan apps that charge huge interest",
        "recognising mobile money fraud", "insurance for farmers and traders", "microfinance and small loans", "paying school fees in instalments", "planning for emergencies with a small fund",
        "sending money to family in the village", "understanding interest rates in plain words"]),
    _d("tourism_hospitality", "Tourism and hospitality", "economy", "commerce culture",
       "visitors, hotels, sites and hosting guests",
       ["visiting a historic town or museum", "eco-tourism and national parks", "running a small guest house", "festival tourism and homecoming visitors",
        "guides and honest service to visitors", "traditional hospitality and welcoming guests", "beaches, waterfalls and hills worth visiting", "street food as tourism",
        "hotels, catering and restaurant careers", "protecting heritage sites"]),
    _d("housing_land", "Housing and land", "economy", "commerce civic numeric risk",
       "building, renting and owning homes",
       ["renting a house: agreements and landlord-tenant respect", "building a house step by step", "mud brick, cement block and roofing materials", "buying land safely and avoiding double sales",
        "the cost of building materials and how to save", "family land and inheritance", "shared compounds and courtyard living", "rent advance and how it strains tenants",
        "urban slums and upgrading", "maintaining a house: drainage, roofing and painting"]),
    _d("consumer_life", "Everyday shopping and consumer life", "economy", "commerce personal civic",
       "buying goods and services for the household",
       ["buying a phone and checking it's genuine", "comparing prices before buying", "second-hand clothes markets", "buying electricity units or credit",
        "warranties and returns", "cooking gas and kerosene purchases", "shopping online safely and delivery riders", "planning monthly household purchases",
        "supermarkets versus open markets", "fake and substandard products"]),
    # ------------------------------------------------------------------ tech & infrastructure
    _d("technology_digital", "Technology and the internet", "tech_infrastructure", "technical commerce academic risk modern_only",
       "computers, phones and the internet in everyday life",
       ["using a smartphone for the first time", "sending and reading email", "keeping passwords and accounts safe", "how the internet works, in simple terms",
        "mobile data bundles and saving data", "video calls and online meetings", "coding clubs and learning to programme", "digital skills for job seekers",
        "misinformation and checking facts on WhatsApp", "e-commerce and delivery apps", "cybercafes and computer literacy", "backing up photos and documents"]),
    _d("ai_data", "Artificial intelligence and data", "tech_infrastructure", "technical academic risk modern_only",
       "AI, data and language technology",
       ["what artificial intelligence is, in plain words", "how speech recognition and voice assistants work", "why African languages need more digital data", "machine translation and its limits",
        "chatbots and how to use them wisely", "privacy and personal data online", "AI in farming and health", "data collection projects by communities",
        "jobs and AI: what changes", "how to spot AI-generated content"]),
    _d("telecom_phones", "Mobile phones and telecommunications", "tech_infrastructure", "technical commerce modern_only",
       "networks, SIM cards and phone use",
       ["registering a SIM card", "why network signal drops and what you can do", "airtime, data plans and bundles", "phone repair and common faults",
        "radio and TV versus phone news", "community radio stations", "charging phones when power is off", "solar phone-charging businesses",
        "text message versus voice call versus WhatsApp", "protecting children's phone use"]),
    _d("energy_power", "Energy and electricity", "tech_infrastructure", "technical numeric commerce academic",
       "how people get power and cook",
       ["the national grid and frequent outages", "solar panels and small home systems", "generators: costs, safety and noise", "clean cooking stoves versus firewood",
        "how a prepaid electricity meter works", "saving electricity at home", "hydro-electric dams and their impact", "biogas from farm waste",
        "mini-grids for rural communities", "gas versus kerosene versus charcoal"]),
    _d("transport_mobility", "Transport and mobility", "tech_infrastructure", "commerce numeric civic",
       "moving people and goods",
       ["motorbike and tricycle taxis: safety and fares", "long-distance buses and motor parks", "road conditions in the rainy season", "ferries and river transport",
        "traffic rules and road signs", "trains and railway revival", "transporting farm produce to markets", "getting a driving licence",
        "ride-hailing apps and drivers", "cycling and walking in towns", "airports and air travel basics", "road safety for children"]),
    _d("engineering_construction", "Engineering and construction", "tech_infrastructure", "technical numeric academic commerce",
       "building things that work",
       ["how a bridge carries loads", "mixing concrete and curing it properly", "electrical wiring safety at home", "plumbing and borehole water systems",
        "surveying land and measuring boundaries", "welding and metal fabrication", "road building and drainage", "machines and simple tools like the lever and pulley",
        "maintaining a car engine", "quality control and building codes"]),
    _d("water_sanitation", "Water and sanitation", "tech_infrastructure", "civic numeric academic rural",
       "getting safe water and managing waste",
       ["boreholes, wells and safe water sources", "treating water at home by boiling and filtering", "public toilets and sanitation", "waste sorting and recycling plastic",
        "gutters and drainage in towns", "water vendors and prices", "protecting rivers from pollution", "community water committees",
        "hygiene at school", "the health cost of unsafe water"]),
    # ------------------------------------------------------------------ environment
    _d("environment_climate", "Environment and climate", "environment", "academic civic numeric",
       "nature, climate change and stewardship",
       ["how climate change affects farmers", "planting trees and stopping deforestation", "erosion and land degradation", "desertification and the Sahel",
        "plastic waste and its effects", "protecting mangroves and wetlands", "harmattan dust and health", "sustainable use of forests and bushmeat",
        "climate-smart farming", "environmental clean-up days", "rising sea levels on the coast", "renewable energy and jobs"]),
    _d("weather_seasons", "Weather and seasons", "environment", "academic numeric rural",
       "rainy and dry seasons and their effects",
       ["the rainy and dry seasons in the forest zone", "harmattan and how people cope with it", "reading the sky for signs of rain", "how a weather forecast is made",
        "planting calendars linked to rainfall", "the effects of late or irregular rains", "flood season along rivers", "heat waves and staying cool",
        "how weather affects markets and transport", "traditional weather sayings and what they mean"]),
    _d("animals_wildlife", "Animals and wildlife", "environment", "academic culture",
       "the animals people live alongside",
       ["elephants, lions and national parks", "monkeys and forest wildlife", "birds and migration", "snakes: staying safe and their role in nature",
        "the role of dogs, cats and working animals", "endangered species such as pangolins", "insects: bees, ants and termites", "tortoise and other animals in traditional stories",
        "fish and river life", "living safely near wildlife"]),
    # ------------------------------------------------------------------ knowledge & science
    _d("natural_science", "Natural science", "knowledge", "technical academic numeric timeless",
       "physics, chemistry, biology and earth science explained for learners",
       ["the water cycle", "photosynthesis and how plants make food", "states of matter and everyday examples", "electricity and circuits with a torch",
        "the human digestive system", "acids, bases and household chemicals", "the solar system and the phases of the moon", "germs and how the body fights them",
        "magnets, light and sound experiments", "rocks, soil types and minerals", "why we have day and night", "simple machines and forces"]),
    _d("mathematics", "Mathematics", "knowledge", "technical academic numeric timeless",
       "numbers, shapes and problem solving",
       ["fractions using yams, bread or oranges", "percentages and discounts in the market", "area and perimeter of a farm plot", "averages and simple statistics from everyday data",
        "ratio and proportion in mixing feed or paint", "geometry: angles and triangles", "measuring time, distance and money", "patterns and number sequences",
        "simple interest on savings", "reading tables and charts", "multiplication tricks and mental arithmetic", "probability with dice and games like ayo/oware"]),
    _d("language_linguistics", "Language and linguistics", "knowledge", "academic culture timeless",
       "how languages work and how they are written",
       ["how tones work in tonal languages", "borrowed words and loanwords", "writing systems and orthography reforms", "dialects and standard forms of a language",
        "multilingualism in everyday life", "mother-tongue education", "how children learn to speak", "greetings and their meanings",
        "naming customs and what names mean", "translation and interpreting as a profession", "language endangerment and revival", "code-switching among speakers"]),
    _d("history", "History", "knowledge", "historical academic culture",
       "the past of West African peoples and places, in general well-known terms",
       ["old West African empires such as Mali and Songhai, in outline", "the Oyo Empire and Yoruba city-states, in outline", "the Kingdom of Dahomey, in outline", "the Asante Kingdom and the Golden Stool, in outline",
        "Benin Kingdom and its bronze art, in outline", "the transatlantic slave trade and its legacy", "the coming of colonial rule and its effects", "the road to independence in Ghana and Nigeria",
        "Kanem-Bornu and the Sokoto Caliphate, in outline", "trade routes across the Sahara", "how oral historians keep the past alive", "ancient towns like Ile-Ife, Kano and Timbuktu"]),
    _d("geography_places", "Geography and places", "knowledge", "academic culture",
       "rivers, regions, cities and landscapes",
       ["the River Niger and River Volta", "vegetation zones from coast to Sahel", "major cities and what they are known for", "lakes, lagoons and coastal features",
        "hills, plateaus and mountains such as the Jos Plateau", "border regions and shared cultures", "the Niger Delta's creeks", "maps: reading directions and scale",
        "population and migration patterns", "climate zones and how they shape farming", "national capitals and why they were chosen", "islands and lagoon towns"]),
    _d("media_journalism", "Media and journalism", "knowledge", "civic academic",
       "news, radio, television and the press",
       ["how a news story is gathered and checked", "community radio and phone-in shows", "fake news and how to check it", "reporters' ethics and the public interest",
        "the role of newspapers in a democracy", "interviewing skills", "television dramas and public discussion", "social media influencers and responsibility",
        "advertising and how to read it critically", "press freedom and citizen journalism"]),
    # ------------------------------------------------------------------ culture & arts
    _d("religion_ethics", "Religion, ethics and values", "culture_arts", "faith culture personal",
       "faith, morality and living well with others",
       ["honesty and integrity in daily life", "forgiveness and reconciliation", "Christian and Muslim festivals and shared neighbourliness", "traditional religion and respect for ancestors",
        "helping the poor and giving to charity", "prayer, fasting and self-discipline", "respect for elders and parents", "the golden rule across faiths",
        "living peacefully with neighbours of another religion", "the value of hard work in faith teachings", "gratitude and contentment", "teaching moral lessons to children"]),
    _d("arts_literature", "Arts and literature", "culture_arts", "culture",
       "writing, painting, theatre and the visual arts",
       ["African writers and why reading matters", "writing a poem or short story", "oral literature: praise poetry and riddles", "sculpture and mask carving traditions",
        "painting and contemporary African art", "theatre and community drama", "book clubs and libraries", "writing in local languages",
        "textile designs and their meanings", "photography and telling stories with images"]),
    _d("music", "Music and dance", "culture_arts", "culture commerce",
       "drumming, songs, popular music and dance",
       ["talking drums and other traditional instruments", "highlife, juju and afrobeats: a light overview", "gospel and church choirs", "Islamic praise singing and the tradition of waka",
        "hip-hop and rap in local languages", "how to learn an instrument", "the role of music at weddings and naming ceremonies", "dance styles and their occasions",
        "radio hits, record shops and streaming", "lullabies and children's songs", "the role of a bandleader and a cultural troupe", "protecting musicians' rights"]),
    _d("film_nollywood", "Film and Nollywood", "culture_arts", "culture commerce",
       "movies, home video and the film industry",
       ["how a Nollywood film is made on a small budget", "why family dramas and comedies are popular", "film in local languages such as Hausa (Kannywood) and Twi", "the role of actors, directors and producers",
        "cinema-going and watching films at home", "writing a screenplay", "film piracy and its harm", "streaming services and African stories",
        "child actors and school drama clubs", "film festivals", "special effects on a low budget", "what makes a good film review"]),
    _d("sports", "Sports and games", "culture_arts", "culture numeric personal",
       "football, athletics and community sport",
       ["football fever and local leagues", "how the football pools of a national team unite people", "athletics: sprinters and long-distance runners", "wrestling and traditional sports such as dambe",
        "school sports day", "how to start a small football club", "fair play and respecting referees", "women in sport",
        "exercise and sport for health", "boxing and its heroes, in general terms", "table tennis, basketball and volleyball", "the Africa Cup of Nations in general terms"]),
    _d("games_pastimes", "Traditional games and pastimes", "culture_arts", "culture personal",
       "play, riddles and leisure",
       ["ayo/oware and other mancala-style board games", "hide and seek and street games children play", "storytelling nights under the moon", "riddles and how to make them",
        "card games and dominoes among adults", "puppet and mask plays", "kite flying and toy making from waste", "video games and phones among children",
        "clapping and skipping rhymes", "family game night"]),
    _d("food_cooking", "Food and cooking", "culture_arts", "culture commerce personal",
       "everyday dishes, ingredients and kitchens",
       ["jollof rice and party food", "soups and stews with local vegetables", "pounded yam, fufu, eba and other swallows", "tuwo, miyan kuka and northern dishes",
        "kenkey, banku and waakye", "street food such as suya, akara and bean cakes", "making palm oil stew or groundnut soup", "fermented foods like ogi, ogiri and dawadawa",
        "preserving food without a fridge", "cooking for a wedding or festival", "cooking on a budget", "traditional drinks such as zobo, kunu and palm wine"]),
    _d("fashion_textiles", "Fashion and textiles", "culture_arts", "culture commerce",
       "clothes, fabrics and personal style",
       ["aso-oke, kente, ankara and other fabrics", "how tailors work with clients", "wearing traditional attire to weddings", "headwraps (gele) and caps",
        "school uniforms and dressing for work", "second-hand clothing and fast fashion", "hair braiding and salon culture", "how to care for and wash fabrics",
        "modern designers using traditional cloth", "what colours and patterns mean in cloth"]),
    _d("festivals_customs", "Festivals and customs", "culture_arts", "culture faith",
       "celebrations, rites of passage and ceremonies",
       ["the New Yam festival", "naming ceremonies for a newborn", "weddings: engagement, dowry and celebration", "funerals and how communities mourn and celebrate a life",
        "Eid, Christmas and Easter in the community", "harvest festivals", "masquerade festivals and their etiquette", "coming-of-age ceremonies",
        "Durbar and horse festivals", "carnival and street parades", "the greeting customs of different peoples", "hospitality: kola nut and drinks for guests"]),
    _d("proverbs_oral_tradition", "Proverbs and oral tradition", "culture_arts", "culture timeless faith",
       "sayings, folktales and spoken heritage",
       ["proverbs about patience and hard work", "proverbs about community and unity", "the role of the tortoise/spider in folktales", "storytelling at night and how stories teach",
        "idioms and figurative expressions in daily speech", "praise names and appellations", "riddles and tongue twisters", "griots and other keepers of memory",
        "proverbs about family and children", "how elders use sayings to advise", "wise sayings about money and honesty", "myths of origin, told as stories people tell"]),
    # ------------------------------------------------------------------ people & life
    _d("family_relationships", "Family and relationships", "people", "personal culture faith",
       "households, marriage, friendships and kin",
       ["the extended family and mutual support", "courtship and choosing a spouse", "in-laws and getting along", "raising children together",
        "friendships and keeping promises", "polygamy and monogamy: family life explained sensitively", "handling family disagreements", "caring for aged parents",
        "neighbours and sharing", "children living with relatives", "keeping family ties across distance", "money and family obligations"]),
    _d("childhood_parenting", "Childhood and parenting", "people", "personal academic risk",
       "growing up and bringing up children",
       ["teaching children good manners", "discipline without violence", "homework help and school routines", "children's chores and responsibility",
        "screen time and phones for children", "talking to teenagers", "bedtime stories and lullabies", "raising a child with a good name and character",
        "child labour and keeping children in school", "birthday celebrations and children's play"]),
    _d("youth_life", "Youth and young adult life", "people", "personal civic",
       "the concerns of young people",
       ["choosing between university and a trade", "peer pressure and making good friends", "social media and self-image", "starting a side hustle as a student",
        "youth in politics and volunteering", "romance and heartbreak, gently told", "moving out and independent living", "dressing and identity",
        "mentors and role models", "staying positive when jobs are scarce"]),
    _d("elders_ageing", "Elders and ageing", "people", "personal culture faith",
       "old age, wisdom and respect",
       ["the respected role of elders in the family", "grandparents telling stories", "retirement and pensions", "caring for elderly parents at home",
        "loneliness in old age and visiting neighbours", "elders' knowledge of herbs and history", "health checks for older people", "passing on skills to the young",
        "ageing gracefully and staying active", "honouring elders at festivals"]),
    _d("urban_life", "Urban life", "people", "civic commerce personal",
       "living in big towns and cities",
       ["moving through traffic in a big city", "living in a rented room or compound in the city", "noise, pollution and small daily pleasures", "the hustle: many jobs to make ends meet",
        "city markets, malls and street vendors", "neighbours you barely know versus village closeness", "eating out and canteens", "city nightlife and staying safe",
        "estates and gated communities", "public spaces, parks and joggers", "getting around a city on a budget", "power cuts and water shortages in city homes"]),
    _d("rural_life", "Rural and village life", "people", "rural culture personal",
       "life in villages and small towns",
       ["a day in the life of a village farmer", "the village square and communal work", "fetching water and firewood", "the weekly rural market",
        "festivals and the return of sons and daughters from the city", "village health centres and clinics", "children walking to school", "village elders and settling disputes",
        "the lack of electricity and how villagers cope", "rural roads and bridges", "sharing a harvest with neighbours", "hunters, farmers and blacksmiths in a village"]),
]

DOMAINS: dict[str, Domain] = {d.key: d for d in _DOMAIN_LIST}
DOMAIN_KEYS: list[str] = [d.key for d in _DOMAIN_LIST]
DOMAIN_GROUPS: list[str] = sorted({d.group for d in _DOMAIN_LIST})


def all_pairs() -> list[tuple[str, str]]:
    """Every (domain key, sub-topic) pair, in stable declaration order."""
    return [(d.key, s) for d in _DOMAIN_LIST for s in d.subtopics]


def domain_tags(domain_key: str) -> frozenset[str]:
    return DOMAINS[domain_key].tags


# ---------------------------------------------------------------------------
# Secondary attribute vocabularies
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Option:
    """One value of a secondary attribute.

    `weight` is the default target share (overridable in the yaml configs).
    `tags_any` / `tags_none` constrain which *domains* the option may be
    paired with (domain tag sets, see module docstring).
    """

    key: str
    text: str  # description shown to the generation model
    weight: float = 1.0
    tags_any: Optional[frozenset[str]] = None  # domain must have at least one of these
    tags_none: frozenset[str] = frozenset()  # domain must have none of these

    def domain_ok(self, tags: frozenset[str]) -> bool:
        if self.tags_any is not None and not (self.tags_any & tags):
            return False
        if self.tags_none & tags:
            return False
        return True


def _fs(s: str) -> frozenset[str]:
    return frozenset(s.split())


@dataclass(frozen=True)
class Genre(Option):
    """A text genre for pretraining documents, with its own compatibility."""

    format_hint: str = "plain running prose, no markdown"
    registers: Optional[frozenset[str]] = None
    lengths: Optional[frozenset[str]] = None
    perspectives: Optional[frozenset[str]] = None
    eras: Optional[frozenset[str]] = None


def _g(key: str, text: str, format_hint: str, registers: str, perspectives: str, *, tags_any: str | None = None,
       tags_none: str = "", lengths: str | None = None, eras: str | None = None) -> Genre:
    return Genre(
        key=key, text=text, format_hint=format_hint,
        tags_any=_fs(tags_any) if tags_any else None, tags_none=_fs(tags_none),
        registers=_fs(registers), perspectives=_fs(perspectives),
        lengths=_fs(lengths) if lengths else None, eras=_fs(eras) if eras else None,
    )


GENRES: dict[str, Genre] = {g.key: g for g in [
    _g("news_report", "a local news report (who/what/where/when/why) about a plausible, non-specific event", "plain prose in short paragraphs; no invented statistics or real people's names",
       "formal semi_formal", "third_person", tags_none="timeless", eras="contemporary"),
    _g("explainer", "an explainer that makes a topic clear to an interested non-expert", "prose; short paragraphs; at most one simple list",
       "formal semi_formal conversational technical", "third_person second_person collective_we"),
    _g("blog_post", "a personal-voice blog post with a clear point of view", "prose in short paragraphs, no markdown",
       "semi_formal conversational colloquial literary", "first_person second_person collective_we"),
    _g("dialogue", "a natural conversation between two or three named speakers", "script style: 'Name: line' on separate lines, no stage directions except very short ones",
       "conversational colloquial semi_formal formal", "dialogic"),
    _g("letter", "a letter (to a relative, friend, official or newspaper editor)", "letter form: greeting, body, closing, sender's name",
       "formal semi_formal conversational", "first_person", tags_none="technical"),
    _g("short_story", "an original short story with characters, a problem and a resolution", "narrative prose; dialogue allowed",
       "literary conversational semi_formal child_friendly", "first_person third_person", lengths="medium long"),
    _g("how_to_guide", "a practical how-to guide with clear ordered steps", "numbered steps as plain lines ('1.', '2.') are allowed; no markdown headings",
       "semi_formal conversational technical formal", "second_person collective_we third_person", tags_none="historical timeless"),
    _g("faq", "an FAQ: several questions people really ask, each with a clear answer", "'Question:' / 'Answer:' pairs in the target language",
       "semi_formal conversational formal", "second_person third_person"),
    _g("encyclopedia_entry", "a neutral encyclopedia-style entry", "prose; an opening definition then organised paragraphs; no markdown headings",
       "formal technical", "third_person", tags_none="personal"),
    _g("opinion_piece", "an opinion piece / editorial that argues a position fairly and respectfully", "prose; a clear thesis and a conclusion",
       "formal semi_formal conversational colloquial", "first_person collective_we"),
    _g("interview", "an interview between a reporter and a (fictional, unnamed or generically named) guest", "'Name: line' turns; a short introduction sentence first",
       "conversational semi_formal formal", "dialogic"),
    _g("speech", "a speech delivered at a community, school, wedding or civic event", "spoken-style prose, with greetings to the audience at the start",
       "formal semi_formal", "first_person collective_we", tags_none="technical"),
    _g("radio_script", "a radio programme segment with a presenter (and possibly a caller)", "script style with speaker labels; a lively spoken tone",
       "conversational semi_formal colloquial", "first_person second_person dialogic collective_we"),
    _g("forum_post", "an online forum or community-group post asking for or giving advice, with one or two replies", "informal prose; replies marked with the replier's name",
       "colloquial conversational", "first_person second_person", lengths="short medium"),
    _g("product_description", "a product or service description for a shop, market stall or app", "short persuasive prose; no invented prices or claims of certification",
       "semi_formal conversational colloquial", "second_person third_person", tags_any="commerce", lengths="short medium", eras="contemporary"),
    _g("lecture_notes", "a teacher's lecture notes or lesson explanation for students", "organised prose with simple numbered points allowed; no markdown headings",
       "formal technical semi_formal", "third_person collective_we", tags_any="academic"),
    _g("folktale", "an ORIGINAL story written in the style of a folktale (do not claim it is an authentic traditional tale)", "narrative prose with a moral at the end",
       "literary conversational child_friendly", "third_person", tags_any="culture", eras="traditional_heritage historical"),
    _g("poem_song_lyrics", "a poem or song lyric (original) with imagery from local life", "short lines/verses as plain text; rhyme optional",
       "literary conversational colloquial", "first_person second_person collective_we third_person", lengths="short medium"),
    _g("personal_narrative", "a first-hand personal narrative of an experience (fictional but realistic)", "narrative prose in the first person",
       "conversational semi_formal literary colloquial", "first_person", tags_none="technical"),
    _g("public_notice", "a public notice or announcement from a school, market association, church/mosque, clinic or council", "short formal notice: heading line, body, contact-free sign-off",
       "formal semi_formal", "third_person collective_we", tags_any="civic commerce academic", lengths="short", eras="contemporary"),
    _g("review", "a review of a film, book, dish, place, product or service", "prose with a clear verdict; no invented star ratings from real organisations",
       "semi_formal conversational colloquial", "first_person", tags_any="commerce culture", lengths="short medium", eras="contemporary"),
    _g("case_study", "a short case study of a (fictional or generic) person, business or community facing a problem", "prose: situation, action, result, lesson",
       "formal semi_formal technical", "third_person", tags_any="commerce academic civic", lengths="medium long"),
    _g("sermon_devotional", "a short sermon, devotional or moral reflection", "spoken-style prose ending with a blessing or a call to reflection",
       "formal semi_formal literary conversational", "second_person collective_we first_person", tags_any="faith"),
    _g("social_media_thread", "a social-media thread (several short connected posts) on the topic", "numbered or separated short posts, each 1-3 sentences",
       "colloquial conversational", "first_person second_person collective_we", lengths="short medium", eras="contemporary"),
    _g("meeting_minutes", "the minutes of a community, school or association meeting (fictional, generic)", "structured lines: attendees (generic titles), items discussed, decisions",
       "formal semi_formal", "third_person", tags_any="civic commerce academic", lengths="short medium", eras="contemporary"),
    _g("textbook_passage", "a passage from a school textbook with definitions and examples", "organised prose; simple numbered examples allowed",
       "formal technical semi_formal", "third_person collective_we", tags_any="academic"),
    _g("advice_column", "an advice column: a reader's short question and the columnist's thoughtful answer", "'Question' paragraph then 'Answer' paragraphs",
       "conversational semi_formal", "second_person third_person", tags_any="personal civic commerce"),
    _g("profile_feature", "a profile of a FICTIONAL, unnamed or generically named community figure (a farmer, nurse, tailor, teacher ...)", "feature prose with scene-setting and a quote-free narrative",
       "semi_formal literary conversational", "third_person first_person", tags_any="personal commerce culture civic"),
]}

REGISTERS: dict[str, Option] = {o.key: o for o in [
    Option("formal", "formal, respectful, standard written language", 1.0),
    Option("semi_formal", "clear everyday written language: polite but not stiff", 1.4),
    Option("conversational", "relaxed spoken-style language, as between friends or neighbours", 1.4),
    Option("colloquial", "informal street/social-media style with natural idioms and light slang (still readable)", 0.8),
    Option("technical", "precise and terminology-aware, but always explained in plain words", 0.6),
    Option("literary", "expressive, imagery-rich language in a storytelling or poetic voice", 0.6),
    Option("child_friendly", "very simple words and short sentences suitable for children", 0.4, tags_none=_fs("risk technical")),
]}

# The register of the *user's request* in SFT/DPO data: a smaller subset.
USER_REGISTERS: dict[str, Option] = {k: REGISTERS[k] for k in ("formal", "semi_formal", "conversational", "colloquial")}

AUDIENCES: dict[str, Option] = {o.key: o for o in [
    Option("general_public", "the general public", 1.6),
    Option("secondary_students", "secondary-school students", 1.0, tags_any=_fs("academic culture civic personal")),
    Option("university_students", "university and college students", 0.7, tags_any=_fs("academic technical civic commerce")),
    Option("children", "children aged about 8-12", 0.5, tags_none=_fs("risk")),
    Option("teenagers", "teenagers", 0.8),
    Option("young_adults", "young adults starting out in work and life", 1.0),
    Option("professionals", "working professionals in the field", 0.6, tags_any=_fs("technical commerce academic civic")),
    Option("small_business_owners", "traders and small-business owners", 0.8, tags_any=_fs("commerce numeric")),
    Option("farmers_rural", "farmers and rural communities", 0.8, tags_any=_fs("rural culture civic")),
    Option("parents", "parents and caregivers", 0.8, tags_any=_fs("personal civic")),
    Option("elders", "older adults", 0.5),
    Option("women_groups", "women's community and market associations", 0.6, tags_any=_fs("personal civic commerce culture")),
    Option("community_leaders", "community leaders, councils and association executives", 0.5, tags_any=_fs("civic commerce culture")),
    Option("newcomers_migrants", "newcomers to a city or country", 0.4, tags_any=_fs("civic commerce personal")),
]}

LENGTH_BUCKETS: dict[str, Option] = {o.key: o for o in [
    Option("short", "120-220 words", 0.35),
    Option("medium", "250-420 words", 0.45),
    Option("long", "450-700 words", 0.20),
]}
LENGTH_WORD_RANGES: dict[str, tuple[int, int]] = {"short": (120, 220), "medium": (250, 420), "long": (450, 700)}

PERSPECTIVES: dict[str, Option] = {o.key: o for o in [
    Option("first_person", "first person (I / we as a person)", 0.25),
    Option("second_person", "second person, addressing the reader directly (you)", 0.15),
    Option("third_person", "third person, neutral or narrating about others", 0.35),
    Option("collective_we", "collective 'we' of a community, class or organisation", 0.15),
    Option("dialogic", "a dialogue between people", 0.10),
]}

ERAS: dict[str, Option] = {o.key: o for o in [
    Option("contemporary", "the present day", 0.60, ),
    Option("traditional_heritage", "traditional life and heritage as still remembered and practised", 0.15, tags_none=_fs("modern_only")),
    Option("historical", "a historical period (in general, well-known terms)", 0.10, tags_none=_fs("modern_only")),
    Option("generational_contrast", "then versus now: how things changed between generations", 0.15, tags_none=_fs("modern_only")),
]}

DIFFICULTIES: dict[str, Option] = {o.key: o for o in [
    Option("basic", "basic reading level: everyday vocabulary, short sentences", 0.40),
    Option("general", "general adult reading level", 0.45),
    Option("advanced", "more advanced: richer vocabulary and reasoning (but stay grammatically safe)", 0.15,
           tags_any=_fs("academic technical civic commerce")),
]}

# Response-length buckets for SFT/DPO answers (words, in the target language).
RESPONSE_LENGTHS: dict[str, Option] = {o.key: o for o in [
    Option("brief", "one short phrase or sentence (up to ~15 words)", 0.25),
    Option("short", "a short paragraph (~30-80 words)", 0.35),
    Option("medium", "a medium answer (~80-180 words)", 0.28),
    Option("long", "a long, well-organised answer (~180-320 words)", 0.12),
]}
RESPONSE_WORD_RANGES: dict[str, tuple[int, int]] = {"brief": (1, 15), "short": (30, 80), "medium": (80, 180), "long": (180, 320)}

INSTRUCTION_STYLES: dict[str, Option] = {o.key: o for o in [
    Option("direct_command", "a direct imperative instruction", 1.0),
    Option("polite_request", "a polite request ('please ...', 'could you ...')", 1.0),
    Option("question_form", "phrased as a question", 1.0),
    Option("terse_keywords", "terse, like a phone-typed message or search query", 0.6),
    Option("context_first", "starts with one sentence of personal context, then the request", 0.9),
    Option("constraint_included", "includes an explicit, checkable constraint (word limit, number of items, format, or required content)", 0.8),
]}


def option_weights(options: dict[str, Option]) -> dict[str, float]:
    return {k: o.weight for k, o in options.items()}


# ---------------------------------------------------------------------------
# Compatibility for the pretraining attribute set
# ---------------------------------------------------------------------------


def pretrain_compat(chosen: dict[str, str], attr: str, value: str) -> bool:
    """Is `value` allowed for `attr`, given the attributes chosen so far?

    Attributes are chosen in the order: genre, register, audience, length_bucket,
    perspective, era, difficulty, locale -- so each rule below may only look
    at attributes chosen *earlier* (the sampler passes exactly those).
    """
    tags = DOMAINS[chosen["domain"]].tags
    genre = GENRES.get(chosen.get("genre", ""))

    if attr == "genre":
        return GENRES[value].domain_ok(tags)
    if attr == "register":
        return genre is None or genre.registers is None or value in genre.registers
    if attr == "audience":
        if not AUDIENCES[value].domain_ok(tags):
            return False
        reg = chosen.get("register")
        if reg == "child_friendly":
            return value in ("children", "teenagers")
        if value == "children":
            return reg in ("conversational", "literary", "semi_formal")
        return True
    if attr == "length_bucket":
        return genre is None or genre.lengths is None or value in genre.lengths
    if attr == "perspective":
        return genre is None or genre.perspectives is None or value in genre.perspectives
    if attr == "era":
        if not ERAS[value].domain_ok(tags):
            return False
        return genre is None or genre.eras is None or value in genre.eras
    if attr == "difficulty":
        if not DIFFICULTIES[value].domain_ok(tags):
            return False
        return not (chosen.get("register") == "child_friendly" and value == "advanced")
    return True  # locale and anything else: unconstrained


def domain_group_of(domain_key: str) -> str:
    return DOMAINS[domain_key].group


# Re-exported for convenience so callers can import everything from taxonomy.
from sampling.locales import COUNTRIES, LOCALES, NAME_POOLS, locale_info, locale_labels  # noqa: E402,F401
