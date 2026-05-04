# Asistente Fiscal con Gemini, RAG y LangGraph

Agente de IA para gestorías españolas que asesora sobre obligaciones fiscales de **autónomos y sociedades**. Responde preguntas sobre cómo rellenar declaraciones, informa de plazos del calendario fiscal 2026 y avisa con la antelación necesaria según la complejidad de cada impuesto.

## Stack tecnológico

| Componente | Tecnología |
|---|---|
| LLM | Google Gemini 2.5 Flash |
| Embeddings | sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2 (local) |
| Base de conocimiento vectorial | ChromaDB |
| Framework de agente | LangGraph + LangChain |
| Entorno de desarrollo | Jupyter Notebook |
| Interfaz (bonus) | Streamlit |

## Estructura del proyecto

```
Projecto-7-IAGenerativa/
├── .env                          # API keys (no subir a GitHub)
├── .gitignore
├── requirements.txt
├── data/
│   ├── calendario_fiscal.csv     # Plazos 2026 extraídos del calendario oficial AEAT
│   ├── obligaciones_perfil.csv   # Mapa de modelos obligatorios por tipo de contribuyente
│   └── manuales/
│       ├── manual_iva_303_2025.pdf
│       ├── manual_renta_100_130_2025_parte1.pdf
│       ├── manual_renta_100_130_2025_parte2.pdf
│       ├── manual_sociedades_200_202_2024.pdf
│       ├── manual_rentaweb_100_2024.pdf
│       ├── manual_sociedadesweb_200_2024.pdf
│       ├── manual_actividades_economicas_111_115.pdf
│       └── leyes/                # Leyes IVA, IRPF e IS como soporte
├── chroma_db/                    # Base vectorial (generada localmente, no en GitHub)
├── notebooks/
│   └── agente_fiscal.ipynb       # Notebook principal entregable
└── PDFs/                         # Documentos originales de referencia
    └── Calendario/
        └── Calendario_del_contribuyente_2026.pdf
```

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

Ejecuta las celdas en orden. La primera vez indexará todos los documentos en ChromaDB (puede tardar unos minutos). Las siguientes ejecuciones cargan la base de datos ya creada.

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
| leyes/ (soporte) | IVA, IRPF, IS | Ambos |

### Estrategia de chunking

- **PDFs**: chunks de 2000 tokens con overlap de 100 tokens (`RecursiveCharacterTextSplitter`). Se eligió un chunk grande para capturar el contexto completo de cada sección fiscal (un chunk pequeño parte explicaciones de casillas o plazos a mitad, perdiendo coherencia).
- **CSVs**: una fila = un chunk (cada obligación fiscal es una unidad semántica completa)
- **Metadatos por chunk**: `perfil`, `modelos`, `trimestre`, `tipo`, `fuente` — permiten filtrado en la recuperación para no mezclar información de autónomos y sociedades

## Diseño del agente

### Grafo LangGraph

```
[recibir_pregunta] → [recuperar_contexto (RAG)] → [generar_respuesta (Gemini)] → END
```

- **recuperar_contexto**: busca en ChromaDB filtrando por `perfil` cuando está definido
- **generar_respuesta**: Gemini con el system prompt + contexto RAG + historial de conversación
- **MemorySaver**: persiste el historial y el perfil del cliente entre turnos

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

**5. Tres ejemplos few-shot en el propio prompt**
Los ejemplos anclan el formato exacto de respuesta y el comportamiento esperado en los casos más frecuentes (perfil conocido, perfil desconocido, información no disponible). Sin ejemplos, el modelo interpreta las instrucciones de forma variable; con ellos, el formato se estabiliza desde la primera respuesta.

**6. temperature=0 en el LLM**
Un agente fiscal debe dar siempre la misma respuesta ante la misma pregunta. Con temperature > 0 existe variabilidad en fechas, porcentajes o nombres de modelos, lo que erosiona la confianza del usuario. Se fijó a 0 para garantizar determinismo total.

## Casos de prueba

El notebook incluye 5 casos documentados en la sección **Demo interactiva**:

| Caso | Pregunta | Demuestra |
|---|---|---|
| 1 | Plazo modelo 303 1T 2026 | Recuperación del calendario fiscal |
| 2 | Casilla 01 del modelo 303 | RAG sobre manual IVA |
| 3 | Obligaciones autónomo 1T | Filtrado por perfil + calendario |
| 4 | Obligaciones sociedad 2T | Cambio de perfil en la misma sesión |
| 5 | ¿Cuándo empezar a preparar? | Memoria de conversación + lógica de antelación |

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
