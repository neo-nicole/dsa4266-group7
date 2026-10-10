import re
from urllib.parse import urlsplit

import tldextract

ext = tldextract.TLDExtract(suffix_list_urls=())  # offline snapshot, same as EDA
URL_CHARS = re.compile(r"[a-zA-Z0-9\-._~:/?#\[\]@!$&'()*+,;=%]*")  # RFC 3986
BAD_PCT = re.compile(r"%(?![0-9a-fA-F]{2})")
LABEL = re.compile(r"[a-z0-9]([a-z0-9-]{0,61}[a-z0-9])?")  # DNS label: no edge '-', <= 63 chars
SCHEME = re.compile(r"[a-zA-Z][a-zA-Z0-9+.\-]*://")  # only at the start, '://' can also appear in queries


def parts(url):
    # splits URL into (hostname, registered domain, port, userinfo, scheme), raises ValueError if unparseable
    s = urlsplit(url if SCHEME.match(url) else "http://" + url)
    host = s.hostname or ""
    e = ext(host)
    reg = ".".join(x for x in (e.domain, e.suffix) if x) or host
    return host, reg, s.port, s.netloc.rpartition("@")[0], s.scheme


def edit_distance(a, b):
    # Levenshtein distance; strip shared prefix/suffix first so long URLs with local edits stay fast
    while a and b and a[0] == b[0]:
        a, b = a[1:], b[1:]
    while a and b and a[-1] == b[-1]:
        a, b = a[:-1], b[:-1]
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def canonical(url):
    # same cleaning as EDA canonicalize_model_url: no scheme, no leading www/www2, no trailing '/'
    url = re.sub(r"^[A-Za-z][A-Za-z0-9+.\-]*://", "", url)
    url = re.sub(r"^www\d*\.", "", url, flags=re.IGNORECASE)
    return re.sub(r"/+$", "", url)


def structure_error(orig, adv):
    # returns "" if adv is a valid URL with the same domain/port/userinfo as orig, else the reason of invalidity
    if not URL_CHARS.fullmatch(adv):
        return "bad character"
    if BAD_PCT.search(adv):
        return "bad % encoding"
    if adv.count("#") > 1:
        return "bad fragment"  # only one '#' allowed (RFC 3986), no dataset URL has two
    try:
        host, reg, *rest = parts(adv)
    except ValueError:
        return "unparseable"
    o_host, o_reg, *o_rest = parts(orig)
    if reg != o_reg:
        return "registered domain changed"
    if rest != o_rest:
        return "port/userinfo/scheme changed"
    if host != o_host and not (len(host) <= 253 and all(LABEL.fullmatch(l) for l in host.split("."))):
        return "invalid hostname"  # only checked if the host was edited
    if canonical(adv) != adv and canonical(orig) == orig:
        return "not canonical"  # cleaning would undo the edit (9 original URLs are already non-canonical)
    return ""


def invalid_reason(orig, adv, k=3):
    # returns "" if adv is a valid adversarial version of orig within k char edits
    if edit_distance(orig, adv) > k:
        return "over budget"
    return structure_error(orig, adv)


if __name__ == "__main__":
    # quick self-test
    o = "login.paypal-x.com/verify?id=1#top"
    ok = ["login.paypal-x.com/verify?id=1#top",  # unchanged
          "logn.paypal-x.com/verify?id=1#top",  # subdomain
          "login2.paypal-x.com/verify?id=1#top",
          "a.login.paypal-x.com/verify?id=1#top",  # new subdomain label
          "login.paypal-x.com/verfy?id=1#top",  # path
          "login.paypal-x.com/verify?id=2#tap",  # query + fragment
          "login.paypal-x.com/verify?id=1#top%20"]  # valid % encoding (3 edits)
    bad = {"login.paypal-y.com/verify?id=1#top": "registered domain changed",
           "login.paypal-x.comx/verify?id=1#top": "registered domain changed",
           "login-.paypal-x.com/verify?id=1#top": "invalid hostname",
           "login..paypal-x.com/verify?id=1#top": "invalid hostname",
           "login.paypal-x.com:8/verify?id=1#top": "port/userinfo/scheme changed",
           "a@login.paypal-x.com/verify?id=1#top": "port/userinfo/scheme changed",
           "login.paypal-x.com/ver fy?id=1#top": "bad character",
           "login.paypal-x.com/verify?id=1#top%1": "bad % encoding",
           "login.paypal-x.com/verify?id=1#to#p": "bad fragment",
           "login.paypal-x.com/verify?id=1#top/": "not canonical",
           "login.paypal-x.com/verify?id=1#topabcd": "over budget"}
    for a in ok:
        assert invalid_reason(o, a) == "", a
    for a, r in bad.items():
        assert invalid_reason(o, a) == r, (a, invalid_reason(o, a))

    assert invalid_reason("ph.gp", "ph.gp/a") == ""  # appending a path to a bare domain
    assert invalid_reason("ph.gp/a", "ph.gp/a#b") == ""  # adding a fragment
    assert invalid_reason("ph.gp", "ph.gp/") == "not canonical"  # trailing '/'
    assert invalid_reason("w.ph.gp/a", "www.ph.gp/a") == "not canonical"  # leading www.
    assert invalid_reason("www4.me/a", "www4.me/b") == ""  # already non-canonical original
    assert invalid_reason("paypal.com", "paypa1.com") == "registered domain changed"
    assert invalid_reason("a.com/u?q=https://b.com", "x.com/u?q=https://b.com") == "registered domain changed"
    assert edit_distance("kitten", "sitting") == 3 and edit_distance("x" * 5000 + "a", "x" * 5000 + "b") == 1
    print("all checks passed")
