"""Guardrail de entrada: PII, prompt injection e política como código (IA Gen aula 06).

É a primeira camada da defesa em profundidade do RAG:
1. **entrada** (este módulo): bloqueia PII, injeção e tópicos proibidos e valida
   o schema `Pergunta`, antes de qualquer chamada ao LLM;
2. **contexto**: as reviews recuperadas entram delimitadas, como dado e não como
   instrução, e as que têm sinal de injeção são marcadas (`detectar_injecao`);
3. **saída**: a resposta tem schema Pydantic e só pode citar `review_id` recuperados.

Detectores por padrão são a primeira camada, nunca a única: têm falso positivo
e falso negativo (notebook 04, Seção 2).
"""
import hashlib
import math
import re

from pydantic import BaseModel, Field, ValidationError

from voc_rag.schemas import MAX_CARACTERES_PERGUNTA, Pergunta
from voc_rag.texto import PADROES_PII


# --- Seção 1: PII com validação aritmética ----------------------------------------------
def validar_cpf(cpf):
    """Confere os dois dígitos verificadores do CPF (aritmética, não regex)."""
    digitos = re.sub(r"\D", "", cpf)
    if len(digitos) != 11 or digitos == digitos[0] * 11:
        return False
    for n_dig in (9, 10):
        soma = sum(int(digitos[i]) * (n_dig + 1 - i) for i in range(n_dig))
        if (soma * 10 % 11) % 10 != int(digitos[n_dig]):
            return False
    return True


def validar_luhn(numero):
    """Algoritmo de Luhn: valida números de cartão de crédito."""
    digitos = [int(d) for d in re.sub(r"\D", "", numero)][::-1]
    if len(digitos) < 13:
        return False
    soma = sum(d if i % 2 == 0 else (d * 2 - 9 if d * 2 > 9 else d * 2)
               for i, d in enumerate(digitos))
    return soma % 10 == 0


# Na entrada o CPF pode vir sem pontuação e há cartão: regex mais larga + validação.
# E-mail, telefone e CEP são os mesmos padrões usados para mascarar a base.
PADROES_PII_ENTRADA = {
    "CARTAO": (re.compile(r"\b(?:\d[ -]?){13,16}\b"), validar_luhn),
    "CPF": (re.compile(r"\b\d{3}\.?\d{3}\.?\d{3}-?\d{2}\b"), validar_cpf),
    "EMAIL": (PADROES_PII["EMAIL"], None),
    "TELEFONE": (PADROES_PII["TELEFONE"], None),
    "CEP": (PADROES_PII["CEP"], None),
}


def detectar_pii(texto):
    """PII confirmada no texto: lista de {tipo, inicio, fim}. O trecho não é devolvido.

    Os tipos validáveis vêm primeiro e "ocupam" o trecho, para um CPF não ser
    contado de novo como telefone. Candidato reprovado na validação não conta.
    """
    achados, ocupado = [], set()
    for tipo, (padrao, validador) in PADROES_PII_ENTRADA.items():
        for m in padrao.finditer(texto):
            posicoes = set(range(m.start(), m.end()))
            if posicoes & ocupado or (validador and not validador(m.group())):
                continue
            ocupado |= posicoes
            achados.append({"tipo": tipo, "inicio": m.start(), "fim": m.end()})
    return sorted(achados, key=lambda a: a["inicio"])


def mascarar_pii_saida(texto):
    """Guardrail de saída: troca cada PII confirmada por [TIPO]. Devolve (texto, tipos)."""
    achados = detectar_pii(texto)
    for a in sorted(achados, key=lambda a: a["inicio"], reverse=True):
        texto = texto[:a["inicio"]] + f"[{a['tipo']}]" + texto[a["fim"]:]
    return texto, sorted({a["tipo"] for a in achados})


def hash_dado(texto):
    """Hash curto para o log de auditoria: o log nunca guarda o texto (LGPD)."""
    return hashlib.sha256(texto.encode("utf-8")).hexdigest()[:12]


# --- Seção 2: prompt injection ------------------------------------------------------------
PADROES_INJECAO = [
    # os padrões da IA Gen aula 06
    (r"ignore\s+(as|todas\s+as|minhas)?\s*instru", "pt: 'ignore as instruções'"),
    (r"desconsidere\s+(tudo|o\s+que|as\s+instru)", "pt: 'desconsidere tudo'"),
    (r"esque[cç]a\s+(tudo|suas\s+regras|o\s+que)", "pt: 'esqueça suas regras'"),
    (r"voc[eê]\s+agora\s+[eé]\b", "pt: redefinição de papel"),
    (r"(mostre|revele|imprima)\s+.{0,30}prompt\s+d[oe]\s+sistema", "pt: exfiltrar o prompt do sistema"),
    (r"ignore\s+(all\s+)?(previous|above|prior)\s+instructions", "en: 'ignore previous instructions'"),
    (r"you\s+are\s+now\b", "en: redefinição de papel"),
    (r"(reveal|print|show)\s+.{0,30}system\s+prompt", "en: exfiltrar o prompt do sistema"),
    (r"responda\s+apenas\s+com?\b.{0,40}(aprovado|autorizado)", "pt: forçar veredito"),
    (r"do\s+anything\s+now|modo\s+dan\b|dan\s+mode", "jailbreak conhecido (DAN)"),
    # específicos deste RAG
    (r"</?\s*(review|evidencia|contexto|system|sistema)\b", "tenta fechar/abrir os delimitadores do contexto"),
    (r"(n[aã]o|sem)\s+(cite|citar|mencione)\s+(as\s+)?(fontes?|reviews?|review_id)", "pede resposta sem citar evidências"),
    (r"(invente|fabrique|crie)\s+(reviews?|cita[cç][oõ]es|review_ids?)", "pede para inventar evidências"),
]
_INJECAO = [(re.compile(p, re.IGNORECASE), rotulo) for p, rotulo in PADROES_INJECAO]


def _entropia(s):
    """Entropia de Shannon em bits por caractere."""
    freq = {c: s.count(c) / len(s) for c in set(s)}
    return -sum(p * math.log2(p) for p in freq.values())


def detectar_injecao(texto):
    """Sinais de prompt injection no texto (lista de rótulos; vazia = nenhum sinal).

    Serve para a pergunta e também para as reviews recuperadas (injeção indireta).
    """
    sinais = [rotulo for regex, rotulo in _INJECAO if regex.search(texto)]
    blocos = [b for b in re.findall(r"[A-Za-z0-9+/=]{40,}", texto) if _entropia(b) > 4.0]
    if blocos:
        sinais.append(f"possível payload ofuscado ({len(blocos)} bloco(s) de alta entropia)")
    return sinais


# --- Seção 3: política como código ---------------------------------------------------------
class Politica(BaseModel):
    """As regras do guardrail como dado: muda-se a política, não o código."""

    versao: str = "1.0"
    max_caracteres: int = MAX_CARACTERES_PERGUNTA
    bloquear_pii: bool = True             # o RAG analisa reviews agregadas, não um cliente
    limite_injecao: int = 1               # nº de sinais a partir do qual bloqueia
    topicos_proibidos: list[str] = Field(default_factory=lambda: [
        "dados do cliente", "dados pessoais do cliente", "cpf do cliente", "endereço do cliente",
        "telefone do cliente", "e-mail do cliente", "email do cliente",
    ])


POLITICA_PADRAO = Politica()


class AvaliacaoEntrada(BaseModel):
    """O veredito do guardrail. Guarda tipos e hash, nunca o dado pessoal."""

    permitida: bool
    motivos: list[str]
    pii: list[str] = Field(default_factory=list)
    sinais_injecao: list[str] = Field(default_factory=list)
    hash_entrada: str
    pergunta: Pergunta | None = None      # só quando permitida


MENSAGEM_BLOQUEIO = ("Não posso processar essa pergunta: ela contém dados pessoais ou "
                     "instruções não permitidas. Reformule perguntando sobre as reviews.")


def avaliar_entrada(texto, filtros=None, politica=POLITICA_PADRAO):
    """Aplica a política à pergunta. Devolve `AvaliacaoEntrada`; só passa adiante se permitida."""
    texto = texto if isinstance(texto, str) else ""
    motivos = []
    if len(texto) > politica.max_caracteres:
        motivos.append(f"pergunta excede {politica.max_caracteres} caracteres")

    pii = sorted({a["tipo"] for a in detectar_pii(texto)})
    if pii and politica.bloquear_pii:
        motivos.append(f"PII na pergunta ({', '.join(pii)})")

    sinais = detectar_injecao(texto)
    if len(sinais) >= politica.limite_injecao:
        motivos.append(f"{len(sinais)} sinal(is) de prompt injection")

    minusculo = texto.lower()
    motivos += [f"tópico proibido: '{t}'" for t in politica.topicos_proibidos if t in minusculo]

    pergunta = None
    if not motivos:
        try:
            pergunta = Pergunta(texto=texto, filtros=filtros or {})
        except ValidationError as erro:
            motivos += [f"schema: {e['loc'][-1] if e['loc'] else ''} {e['msg']}".strip()
                        for e in erro.errors()]

    return AvaliacaoEntrada(permitida=not motivos, motivos=motivos or ["nenhuma regra acionada"],
                            pii=pii, sinais_injecao=sinais, hash_entrada=hash_dado(texto),
                            pergunta=pergunta)
