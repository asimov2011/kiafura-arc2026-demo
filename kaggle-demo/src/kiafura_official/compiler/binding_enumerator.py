"""Enumerazione dei binding degli slot parametrici.

L'organo mancante fra anti-unificazione e transfer.

L'anti-unificatore produce graph parametrici con slot; finora gli slot
venivano riempiti con ``_mode_default()``, cioè la moda delle costanti
osservate nei **programmi Hodel sorgente**. Un graph costruito così
ricostruisce il proprio task sorgente e non può adattarsi a un task
nuovo, che ha altri colori e altre dimensioni.

Questo modulo sostituisce quel default con l'enumerazione usata dalla
memoria certificata A100, il cui contratto dichiara
``"binding_source": "demonstrations"``: i valori candidati per uno slot
si derivano dalle **dimostrazioni del task in esame**, mai dalla sua
identità.

Due sorgenti legittime, e nessun'altra:

1. **Le dimostrazioni del task** — per gli slot percettivi (COLOR, INT,
   VECTOR, POINT, GRID): colori presenti, background, dimensioni,
   conteggi di componenti. È ciò che il runtime B900 certificato vedeva
   dei task B.
2. **Il vocabolario degli operatori** — per gli slot FUNCTION: i nomi di
   primitiva osservati nel corpus sorgente. È memoria compilata offline
   (Anello 2), l'analogo dei 112 operatori delle 19 famiglie A100, e non
   contiene nulla del task bersaglio.

Vietato per costruzione: task ID, griglie letterali, fingerprint di
input, programmi completi, binding derivati dal target.

Misurato su A2 (81 pattern, 217 slot):

    FUNCTION 53.0%   BOOL 30.4%   COLOR 9.2%
    INT 2.8%   VECTOR 2.3%   ANY/TUPLE/POINT 2.3%

    slot per graph: mediana 3, max 8
    spazio candidati: mediana 80, 72/81 graph sotto 10^6

Vedi ``docs/audit-claude-agosto.md § 4``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Mapping, Sequence

from schema.typed_graph_ir import Constant, ParamSlot, ValueType

Grid = tuple[tuple[int, ...], ...]
Demo = tuple[Grid, Grid]

#: Oltre questo numero di candidati per graph ci si astiene invece di
#: enumerare. Fail-closed: un budget superato è un'astensione, mai un
#: output (AGENTS.md § 6).
DEFAULT_CANDIDATE_BUDGET = 2_000_000

#: Cardinalità massima per singolo slot percettivo. Contiene l'esplosione
#: quando una griglia contiene molti colori o è molto grande.
MAX_VALUES_PER_SLOT = 32


class BindingEnumerationError(RuntimeError):
    """Enumerazione impossibile: si astiene, non si indovina."""


# ──────────────────────────────────────────────────────────────────────
# Feature estratte dalle dimostrazioni
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class DemoFeatures:
    """Il vocabolario percettivo di un task, estratto dai soli demo.

    Ogni campo corrisponde a una feature che le 19 famiglie certificate
    usano nei rispettivi ``candidates(demos)``. Nessun campo dipende
    dall'identità del task.
    """

    colors_all: tuple[int, ...]
    colors_input: tuple[int, ...]
    colors_output: tuple[int, ...]
    colors_added: tuple[int, ...]
    colors_removed: tuple[int, ...]
    backgrounds: tuple[int, ...]
    heights: tuple[int, ...]
    widths: tuple[int, ...]
    dims: tuple[tuple[int, int], ...]
    counts: tuple[int, ...]

    @property
    def small_ints(self) -> tuple[int, ...]:
        """Interi plausibili: dimensioni, conteggi e i piccoli naturali."""
        values = set(self.heights) | set(self.widths) | set(self.counts)
        values |= {0, 1, 2, 3}
        return tuple(sorted(v for v in values if 0 <= v <= 64))


def _grid_colors(grid: Grid) -> set[int]:
    return {cell for row in grid for cell in row}


def _background(grid: Grid) -> int:
    counter = Counter(cell for row in grid for cell in row)
    return counter.most_common(1)[0][0] if counter else 0


def extract_features(demos: Sequence[Demo]) -> DemoFeatures:
    """Estrae il vocabolario percettivo dalle sole dimostrazioni."""
    if not demos:
        raise BindingEnumerationError("nessuna dimostrazione: astensione")

    colors_in: set[int] = set()
    colors_out: set[int] = set()
    backgrounds: set[int] = set()
    heights: set[int] = set()
    widths: set[int] = set()
    dims: set[tuple[int, int]] = set()
    counts: set[int] = set()

    for source, target in demos:
        if not source or not target:
            raise BindingEnumerationError("dimostrazione con griglia vuota")
        colors_in |= _grid_colors(source)
        colors_out |= _grid_colors(target)
        backgrounds.add(_background(source))
        for grid in (source, target):
            heights.add(len(grid))
            widths.add(len(grid[0]))
            dims.add((len(grid), len(grid[0])))
        counts.add(len(_grid_colors(source)))
        counts.add(len(_grid_colors(target)))

    return DemoFeatures(
        colors_all=tuple(sorted(colors_in | colors_out)),
        colors_input=tuple(sorted(colors_in)),
        colors_output=tuple(sorted(colors_out)),
        colors_added=tuple(sorted(colors_out - colors_in)),
        colors_removed=tuple(sorted(colors_in - colors_out)),
        backgrounds=tuple(sorted(backgrounds)),
        heights=tuple(sorted(heights)),
        widths=tuple(sorted(widths)),
        dims=tuple(sorted(dims)),
        counts=tuple(sorted(counts)),
    )


# ──────────────────────────────────────────────────────────────────────
# Vocabolario degli operatori (memoria, non target)
# ──────────────────────────────────────────────────────────────────────


@dataclass(frozen=True)
class FunctionVocabulary:
    """Nomi di primitiva osservati nel corpus sorgente.

    È contenuto di memoria compilato offline: cresce con le coorti
    industrializzate e non guarda mai il task bersaglio. Su A2 conta 26
    voci su 121 primitive disponibili nel registro — ed è questa
    differenza che tiene lo spazio dei candidati trattabile.
    """

    names: tuple[str, ...]

    @classmethod
    def from_slots(cls, slots: Iterable[ParamSlot]) -> "FunctionVocabulary":
        observed = {
            slot.default.value
            for slot in slots
            if slot.value_type is ValueType.FUNCTION
            and slot.default is not None
            and isinstance(slot.default.value, str)
        }
        return cls(names=tuple(sorted(observed)))

    def __len__(self) -> int:
        return len(self.names)


# ──────────────────────────────────────────────────────────────────────
# Enumerazione per slot
# ──────────────────────────────────────────────────────────────────────


def enumerate_slot(
    slot: ParamSlot,
    features: DemoFeatures,
    vocabulary: FunctionVocabulary,
) -> tuple[Constant, ...]:
    """Valori candidati per uno slot, derivati da demo e vocabolario.

    Il default sorgente non è mai usato come valore: al più ordina i
    candidati, mettendo per primo quello già visto nella sorgente, così
    che il task sorgente si ricostruisca per primo senza che il valore
    sia inciso nel graph.
    """
    vtype = slot.value_type

    if vtype is ValueType.BOOL:
        values: list[object] = [False, True]
    elif vtype is ValueType.FUNCTION:
        values = list(vocabulary.names)
    elif vtype is ValueType.COLOR:
        values = list(
            dict.fromkeys(
                [*features.backgrounds, *features.colors_all,
                 *features.colors_added, *features.colors_removed]
            )
        )
    elif vtype is ValueType.INT:
        values = list(features.small_ints)
    elif vtype in (ValueType.VECTOR, ValueType.POINT):
        values = [
            (di, dj)
            for di in (-1, 0, 1)
            for dj in (-1, 0, 1)
            if (di, dj) != (0, 0)
        ]
        values += [(h, w) for h, w in features.dims]
    else:
        # ANY, TUPLE, GRID, OBJECT…: non enumerabili in modo sano.
        # Fail-closed: nessun valore inventato, il chiamante si astiene.
        raise BindingEnumerationError(
            f"slot {slot.name!r} di tipo {vtype.name}: enumerazione non definita"
        )

    if not values:
        raise BindingEnumerationError(
            f"slot {slot.name!r} di tipo {vtype.name}: nessun candidato dai demo"
        )

    # Il valore sorgente, se presente, va in testa: priorità, non identità.
    if slot.default is not None and slot.default.value in values:
        values.remove(slot.default.value)
        values.insert(0, slot.default.value)

    return tuple(
        Constant(type=vtype, value=v) for v in values[:MAX_VALUES_PER_SLOT]
    )


@dataclass
class EnumerationPlan:
    """Piano di enumerazione per un graph: slot, candidati, dimensione."""

    per_slot: dict[str, tuple[Constant, ...]] = field(default_factory=dict)
    skipped: tuple[str, ...] = ()

    @property
    def size(self) -> int:
        total = 1
        for values in self.per_slot.values():
            total *= len(values)
        return total


def plan_for_graph(
    slots: Sequence[ParamSlot],
    demos: Sequence[Demo],
    vocabulary: FunctionVocabulary,
    *,
    budget: int = DEFAULT_CANDIDATE_BUDGET,
) -> EnumerationPlan:
    """Costruisce il piano di enumerazione di un graph, o si astiene.

    Solleva ``BindingEnumerationError`` se un tipo non è enumerabile o se
    lo spazio supera il budget: in entrambi i casi il chiamante deve
    astenersi, mai emettere.
    """
    features = extract_features(demos)
    plan = EnumerationPlan()
    for slot in slots:
        plan.per_slot[slot.name] = enumerate_slot(slot, features, vocabulary)
    if plan.size > budget:
        raise BindingEnumerationError(
            f"spazio candidati {plan.size} oltre il budget {budget}: astensione"
        )
    return plan


def iter_bindings(plan: EnumerationPlan) -> Iterable[Mapping[str, Constant]]:
    """Itera i binding del piano in ordine deterministico.

    L'ordine è quello degli slot ordinati per nome e dei candidati come
    prodotti da ``enumerate_slot``: stessi input → stessa sequenza
    (AGENTS.md § 5).
    """
    names = sorted(plan.per_slot)
    if not names:
        yield {}
        return

    def rec(idx: int, acc: dict[str, Constant]):
        if idx == len(names):
            yield dict(acc)
            return
        name = names[idx]
        for value in plan.per_slot[name]:
            acc[name] = value
            yield from rec(idx + 1, acc)
        acc.pop(name, None)

    yield from rec(0, {})


# ──────────────────────────────────────────────────────────────────────
# Gate anti-copia
# ──────────────────────────────────────────────────────────────────────


def residual_baked_constants(slots: Sequence[ParamSlot]) -> tuple[str, ...]:
    """Slot il cui tipo non è enumerabile dai demo: costanti incise.

    È il gate che distingue un 100/100 **scoperto** da uno **copiato**.
    Un graph ammesso in memoria deve restituire una tupla vuota: ogni suo
    slot deve poter essere ri-derivato dalle dimostrazioni di un task
    nuovo. Il conteggio è meccanico, non richiede revisione umana.
    """
    enumerable = {
        ValueType.BOOL,
        ValueType.FUNCTION,
        ValueType.COLOR,
        ValueType.INT,
        ValueType.VECTOR,
        ValueType.POINT,
    }
    return tuple(
        slot.name for slot in slots if slot.value_type not in enumerable
    )
