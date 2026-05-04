# Asistente Fiscal con Gemini, RAG y LangGraph

Agente de IA para gestorías españolas que asesora sobre obligaciones fiscales de **autónomos y sociedades**. Responde preguntas sobre cómo rellenar declaraciones, informa de plazos del calendario fiscal 2026 y avisa con la antelación necesaria según la complejidad de cada impuesto.

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local, HuggingFace) |
| Base de conocimiento vectorial | ChromaDB |
| Framework de agente | LangGraph + LangChain |
| Entorno de desarrollo | Jupyter Notebook |
| Interfaz (bonus) | Streamlit — desplegada en Streamlit Cloud |

> **Nota sobre embeddings:** Se usa el modelo multilingüe de HuggingFace en lugar de Gemini Embeddings por dos razones: (1) los documentos fiscales incluyen texto en catalán y valenciano extraído de los PDFs de la AEAT, y el modelo `paraphrase-multilingual-MiniLM-L12-v2` maneja mejor la mezcla de idiomas; (2) al ejecutarse localmente elimina una dependencia de API externa en la fase de indexación, reduciendo costes y latencia.

---

## Estructura del proyecto

```
Projecto-7-IAGenerativa/
├── .env                              # API keys (no subir a GitHub)
├── .gitignore
├── requirements.txt
├── app.py                            # Interfaz Streamlit (bonus)
├── data/
│   ├── calendario_fiscal.csv         # Plazos 2026 extraídos del calendario oficial AEAT
│   ├── obligaciones_perfil.csv       # Mapa de modelos obligatorios por tipo de contribuyente
│   └── manuales/
│       ├── practicos/es/             # Manuales prácticos AEAT (PDFs)
│       │   ├── manual_iva_303_2025.pdf
│       │   ├── manual_renta_100_130_2025_parte1.pdf
│       │   ├── manual_renta_100_130_2025_parte2.pdf
│       │   ├── manual_sociedades_200_202_2024.pdf
│       │   └── manual_actividades_economicas_111_115.pdf
│       ├── web/es/                   # Manuales de aplicaciones web AEAT
│       │   ├── manual_rentaweb_100_2024.pdf
│       │   └── manual_sociedadesweb_200_2024.pdf
│       └── leyes/                    # Leyes IVA, IRPF e IS como soporte
├── chroma_db/                        # Base vectorial persistida (incluida en el repo)
├── notebooks/
│   └── agente_fiscal.ipynb           # Notebook principal entregable
└── PDFs/                             # Documentos originales de referencia
```

---

## Instalación

### 1. Clonar el repositorio

```bash
git clone <url-del-repositorio>
cd Projecto-7-IAGenerativa
```

### 2. Crear entorno virtual e instalar dependencias

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

### 3. Configurar la API key de Google Gemini

Crea un archivo `.env` en la raíz del proyecto:

```
GOOGLE_API_KEY=tu_api_key_de_gemini
```

Obtén tu API key en [Google AI Studio](https://aistudio.google.com/app/apikey).

### 4. Ejecutar el notebook

```bash
jupyter notebook notebooks/agente_fiscal.ipynb
```

Ejecuta las celdas en orden. La base vectorial (`chroma_db/`) ya está incluida en el repositorio, por lo que la indexación se omite automáticamente y el agente carga directamente desde disco.

### 5. (Opcional) Ejecutar la interfaz Streamlit en local

```bash
streamlit run app.py
```

---

## Base de conocimiento

### Documentos indexados

| Archivo | Modelos cubiertos | Perfil |
|---|---|---|
| manual_iva_303_2025.pdf | 303 | Ambos |
| manual_renta_100_130_2025_parte1.pdf | 100, 130 | Autónomo |
| manual_renta_100_130_2025_parte2.pdf | 100, 130 | Autónomo |
| manual_sociedades_200_202_2024.pdf | 200, 202 | Sociedad |
| manual_rentaweb_100_2024.pdf | 100 | Autónomo |
| manual_sociedadesweb_200_2024.pdf | 200 | Sociedad |
| manual_actividades_economicas_111_115.pdf | 111, 115 | Ambos |
| calendario_fiscal.csv | Todos | Ambos |
| obligaciones_perfil.csv | Todos | Ambos |

### Estrategia de chunking

- **PDFs:** `SemanticChunker` con umbral percentil 95 — divide respetando fronteras conceptuales (artículos, apartados) en lugar de cortar por número fijo de caracteres. En documentos fiscales, un apartado completo es la unidad mínima de información coherente; cortar a mitad de una explicación de casilla o plazo hace que el modelo recupere contexto incompleto. Se incluye `RecursiveCharacterTextSplitter` (2000 chars / overlap 100) como fallback para textos muy cortos donde el chunker semántico no puede actuar.
- **CSVs:** una fila = un chunk. Cada obligación fiscal es una unidad semántica completa con todos sus campos (modelo, perfil, fecha, periodicidad).
- **Metadatos por chunk:** `perfil`, `modelos`, `trimestre`, `tipo`, `fuente` — permiten filtrado preciso en la recuperación para no mezclar información de autónomos y sociedades.

---

## Diseño del agente

### Arquitectura del grafo LangGraph

```
START
  │
  ▼
podar_historial       ← elimina mensajes si historial > 10 (RemoveMessage)
  │
  ▼
clasificar_consulta   ← detecta tipo de pregunta por keywords
  │
  ├─► recuperar_plazos      (k=10 CSV + k=3 PDF) — preguntas de fechas y plazos
  ├─► recuperar_documentos  (k=8 PDF + k=3 CSV)  — preguntas de cumplimentación
  └─► recuperar_general     (k=5 PDF + k=6 CSV)  — mezcla balanceada
         │
         ▼
    generar_respuesta  ← Gemini 2.5 Flash + contexto RAG + historial
         │
         ▼
        END
```

- **Routing condicional:** la pregunta se clasifica por keywords antes de recuperar contexto. Las preguntas de plazos priorizan los CSVs del calendario; las de cumplimentación priorizan los manuales PDF; el resto usa una mezcla balanceada.
- **Gestión de tokens:** `podar_historial` elimina mensajes en pares cuando el historial supera 10 mensajes, evitando desbordamiento del contexto en conversaciones largas.
- **Moderación en cascada:** antes de llegar al agente, cada pregunta pasa por tres capas — regex de keywords (gratis), clasificador ML TF-IDF (rápido, umbral 85% de confianza) y LLM (solo para casos ambiguos). Las preguntas fuera de ámbito se rechazan sin invocar el agente.
- **MemorySaver:** persiste el historial y el perfil del cliente entre turnos mediante `thread_id`.

### System prompt — justificación

El system prompt se diseñó con seis decisiones concretas, cada una con una razón específica:

**1. Identificación de perfil obligatoria antes de responder**
Las obligaciones de un autónomo y una sociedad son completamente distintas (el autónomo presenta el 130, la sociedad el 202; los plazos del IS no aplican a autónomos, etc.). Sin saber el perfil, cualquier respuesta es potencialmente errónea. Se instruyó al agente para que pregunte explícitamente si el perfil no está claro en la pregunta o en el historial, y para que no asuma.

**2. Regla de oro anti-alucinación con formulación estricta**
En materia fiscal, una fecha o casilla incorrecta puede derivar en sanciones económicas para el cliente. El prompt prohíbe explícitamente inferir o inventar datos que no aparezcan literalmente en el contexto RAG, y define la frase exacta que debe usar el agente cuando no tiene información suficiente. Esto es más robusto que decir simplemente "basa tus respuestas en el contexto".

**3. Estructura de respuesta fija (perfil → obligaciones → plazos → inicio preparación → fuente)**
Sin una estructura definida, el modelo varía el orden y el formato en cada respuesta, lo que dificulta la lectura y la comparación entre consultas. Fijar el orden garantiza consistencia y facilita que el gestor encuentre rápidamente el dato que busca.

**4. Lógica de antelación como propuesta de valor**
El agente no solo informa del plazo límite: calcula la fecha recomendada de inicio de preparación restando `dias_preparacion_recomendados` del calendario. Esta es la funcionalidad diferencial para una gestoría — avisar con suficiente margen según la complejidad de cada modelo.

**5. Five-shot examples en el propio prompt**
Los cinco ejemplos anclan el formato exacto de respuesta y el comportamiento esperado en los casos más frecuentes: perfil conocido con plazo, perfil desconocido, cumplimentación de casilla, obligaciones de un trimestre e información no disponible. Sin ejemplos, el modelo interpreta las instrucciones de forma variable; con ellos, el formato se estabiliza desde la primera respuesta.

**6. temperature=0 en el LLM**
Un agente fiscal debe dar siempre la misma respuesta ante la misma pregunta. Con temperature > 0 existe variabilidad en fechas, porcentajes o nombres de modelos, lo que erosiona la confianza del usuario. Se fijó a 0 para garantizar determinismo total.

---

## Casos de prueba

El notebook incluye 5 casos documentados en la sección **Demo interactiva** (sección 9):

| Caso | Pregunta | Demuestra |
|---|---|---|
| 1 | Plazo modelo 303 1T 2026 | Recuperación del calendario fiscal (router: plazos) |
| 2 | Casilla 01 del modelo 303 | RAG sobre manual IVA (router: documentos) |
| 3 | Obligaciones autónomo 1T | Filtrado por perfil + calendario |
| 4 | Obligaciones sociedad 2T | Cambio de perfil en la misma sesión |
| 5 | ¿Cuándo empezar a preparar? | Memoria de conversación + lógica de antelación |

---

## Modelos fiscales cubiertos

| Modelo | Nombre | Perfil | Periodicidad |
|---|---|---|---|
| 100 | IRPF anual | Autónomo | Anual |
| 111 | Retenciones trabajo/profesionales | Ambos | Trimestral |
| 115 | Retenciones alquileres | Ambos | Trimestral |
| 130 | Pago fraccionado IRPF | Autónomo | Trimestral |
| 200 | Impuesto sobre Sociedades | Sociedad | Anual |
| 202 | Pagos fraccionados IS | Sociedad | Cuatrimestral |
| 303 | IVA trimestral | Ambos | Trimestral |
| 347 | Operaciones con terceros | Ambos | Anual |
| 390 | Resumen anual IVA | Ambos | Anual |
