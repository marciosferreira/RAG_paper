import os
import numpy as np
from functools import wraps

tracing_using_phoenix_arize = True

if tracing_using_phoenix_arize:
    os.environ["OTEL_EXPORTER_OTLP_HEADERS"] = "api_key=04d8476a4c5d41aa628:cbdb4a8"
    os.environ["PHOENIX_CLIENT_HEADERS"] = "api_key=04d8476a4c5d41aa628:cbdb4a8"
    os.environ["PHOENIX_COLLECTOR_ENDPOINT"] = "https://app.phoenix.arize.com"

    from phoenix.otel import register
    tracer_provider = register(project_name="gen_ai")  
    from openinference.semconv.trace import SpanAttributes    
    from opentelemetry import trace

    tracer = trace.get_tracer("gen_ai-llm")

    def phoenix_trace_function(span_name, 
                               kind="None", 
                               input_key="None", 
                               output_key="None", 
                               embeddings_key="None", 
                               metadata_key="adit_metadata"):
        """
        Create the decorator in order to track functions using Phoenix e OpenTelemetry.
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                with tracer.start_as_current_span(span_name) as span:

                    result = func(*args, **kwargs)

                    # Obter o estado (primeiro argumento esperado)
                    state = args[0] if len(args) > 0 else {}                    

                    
                    #registar texto das embeddings resultantes
                    span.set_attribute(SpanAttributes.OPENINFERENCE_SPAN_KIND, kind) # see here for a list of span kinds: https://github.com/Arize-ai/openinference/blob/main/python/openinference-semantic-conventions/src/openinference/semconv/trace/__init__.py#L271



                    embeddings_original = result["resulting_chunk_context"]


                    # Transformação para a estrutura achatada com vetores [0.0]
                    embeddings = [
                        {
                            "embedding.vector": [0.0],  # Preenchendo com [0.0]
                            "embedding.text": item["content"]
                        }
                        for item in embeddings_original
                    ]
          
            
                    # Registrar cada embedding de forma achatada
                    for i, embedding in enumerate(embeddings):
                        for key, value in embedding.items():
                            # Adicionar o índice para evitar conflito de chaves
                            span.set_attribute(f"embedding.embeddings.{i}.{key}", value)
 

                    # Registrar o valor de entrada
                    input_value = state.get(input_key, "Input absent")
                    span.set_attribute(SpanAttributes.INPUT_VALUE, input_value)      
           
                    
                    keys = output_key.split(".")  # Divide a chave em partes
                    current_value = result  # Inicializa com o dicionário principal
                    
                    for key in keys:
                        if isinstance(current_value, dict):  # Verifica se ainda estamos navegando em um dicionário
                            current_value = current_value.get(key, "Output absent")
                        else:
                            current_value = "Output absent"  # Se não for mais um dicionário, interrompe
                            break                    
                    
                    # Caso especial: se for "resulting_context.text", converte o valor final para string
                    if output_key == "resulting_chunk_context" and current_value != "Output absent":
                        output_value = str(current_value)
                    
                    elif output_key == "query_template" and current_value != "Output absent":
                        output_value = str(current_value[0]["content"])                        
                    else:
                        output_value = current_value   

                    span.set_attribute(SpanAttributes.OUTPUT_VALUE, output_value)   
                               
                    additional_metadata = result[metadata_key]
                    # Inserir os pares chave-valor em metadata
                    for key, value in additional_metadata.items():
                        span.set_attribute(f"{SpanAttributes.METADATA}.{key}", value)


                    return result
            return wrapper
        return decorator
else:
    def trace_function(span_name, kind="LLM", input_key="query", output_key="llm_response.content"):
        """
        Fake decorator in case trackihg is deactivated (avoid errors).
        """
        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                print(f"[INFO] tracing using phoenix arize is set to False. Function '{func.__name__}' is running without tracing.")
                return func(*args, **kwargs)
            return wrapper
        return decorator
#check how to create attributes: https://docs.arize.com/arize/llm-tracing/how-to-tracing-manual/instrumenting-span-types