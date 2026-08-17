from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.sdk.resources import Resource

# Initialize OpenTelemetry Tracer with Service Resource Name
_resource = Resource.create({"service.name": "enterprise-rag-engine"})
_provider = TracerProvider(resource=_resource)

# Export traces to console in JSON format
_processor = BatchSpanProcessor(ConsoleSpanExporter())
_provider.add_span_processor(_processor)

trace.set_tracer_provider(_provider)
_tracer = trace.get_tracer("enterprise.rag.tracer")

def get_tracer():
    return _tracer