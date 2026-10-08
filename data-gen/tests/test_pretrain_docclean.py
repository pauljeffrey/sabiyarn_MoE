from pretrain_new import docclean as D

GOOD = ("Wannan jimla ce ta gwaji domin tabbatar da cewa tsarin tsaftace rubutu yana aiki yadda ya kamata. " * 3 + "\n"
        "Ana amfani da wannan rubutu domin gwada ingancin tsarin tsaftacewa da kuma cire kwafi a cikin bayanai da yawa.\n"
        "Kowace sakin layi tana da kalmomi da yawa domin ta wuce mafi ƙarancin tsayin da aka tsara don daftarin aiki.")


def test_keeps_a_normal_document_and_drops_menu_debris():
    t = D.clean_document("Home\nAbout\nContact us\n" + GOOD + "\nhttps://spam.example.com/x\nCopyright 2025 all rights reserved", "hau")
    assert t is not None and t.startswith("Wannan") and "spam.example" not in t and "Copyright" not in t


def test_short_documents_are_dropped_with_language_dependent_limits():
    assert D.clean_document("Gajeren rubutu ne kawai.", "hau") is None
    mid = " ".join(["kalma"] * 8 + ["wata", "kuma", "gida", "ruwa", "hanya", "mutum", "dare", "rana"] * 3)
    assert D.clean_document(mid + ".", "hau") is None and D.clean_document(mid + ".", "twi") is not None


def test_rejects_digit_heavy_and_repetitive_text():
    assert D.clean_document(" ".join(["1234567"] * 80), "hau") is None
    assert D.clean_document(" ".join(["abeg"] * 200) + ".", "pcm") is None


def test_lines_with_contact_details_are_removed():
    t = D.clean_document(GOOD + "\nKira mu a +234 801 234 5678 ko aika wasiku zuwa info@example.com domin karin bayani", "hau")
    assert t is not None and "example.com" not in t and "801" not in t


def test_deduper_drops_exact_and_near_duplicates_but_keeps_distinct_documents():
    para = lambda i: (f"Wannan sakin layi ta {i} tana ɗauke da isassun kalmomi daban-daban domin ta kasance ta musamman a cikin bayanan. " * 2)
    a = "\n".join(para(i) for i in range(6))
    b = "\n".join(para(i) for i in range(3, 9))        # half the paragraphs shared with a
    c = "\n".join(para(i) for i in range(20, 26))
    d = D.Deduper(bits=18)
    assert not d.is_duplicate(a)
    assert d.is_duplicate(a) and d.dropped_exact == 1
    assert d.is_duplicate(b) and d.dropped_near == 1
    assert not d.is_duplicate(c)
