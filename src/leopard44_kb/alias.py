"""Deterministic, privacy-preserving aliases for message senders.

Until the product runs privately on each owner's own machine — where an owner may
legitimately see their own contacts' real names — any sender surfaced in a SHARED
or HOSTED context (the public alpha, the ``#L44alpha`` feedback collation, any
community-layer attribution) is shown as an *interesting random* alias instead of
a real name, phone number, or WhatsApp JID/LID.

The alias is a pure function of the sender identifier, so the same person always
maps to the same alias: distinct senders stay distinguishable across messages,
but no real identity is exposed. Swap the alias for a real name only in the
private, per-owner build.
"""
from __future__ import annotations

import hashlib

# Marine / sailing themed — fits the Leopard 44 audience and reads as friendly.
_ADJECTIVES = (
    "Salty", "Windward", "Leeward", "Drifting", "Anchored", "Breezy", "Tidal",
    "Coastal", "Nimble", "Brass", "Teak", "Copper", "Foggy", "Cresting", "Rolling",
    "Steady", "Jolly", "Weathered", "Spirited", "Curious", "Crafty", "Hardy",
    "Plucky", "Briny", "Gusty", "Mellow", "Barnacled", "Sunlit", "Stormy",
    "Trusty", "Wandering", "Lucky", "Brave", "Quiet", "Restless", "Seaworthy",
)
_NOUNS = (
    "Albatross", "Petrel", "Mariner", "Skipper", "Dolphin", "Marlin", "Tern",
    "Gull", "Nautilus", "Compass", "Halyard", "Winch", "Rudder", "Spinnaker",
    "Tradewind", "Helmsman", "Bosun", "Navigator", "Stingray", "Seahorse",
    "Pelican", "Frigate", "Schooner", "Catamaran", "Rigger", "Keel", "Anchor",
    "Beacon", "Current", "Dinghy", "Galley", "Jib", "Lighthouse", "Mast", "Tiller",
    "Reef",
)


def sender_alias(identifier: str | None) -> str:
    """Map a sender identifier (JID / phone / LID) to a stable, interesting alias.

    Same identifier → same alias; different identifiers almost always differ
    (36 × 36 × 100 ≈ 130k combinations). Empty/unknown senders → "Unknown Sailor".

    >>> sender_alias("64472190349336@lid") == sender_alias("64472190349336@lid")
    True
    """
    if not identifier:
        return "Unknown Sailor"
    h = hashlib.sha256(identifier.encode("utf-8")).digest()
    adjective = _ADJECTIVES[h[0] % len(_ADJECTIVES)]
    noun = _NOUNS[h[1] % len(_NOUNS)]
    number = h[2] % 100
    return f"{adjective} {noun} {number:02d}"
