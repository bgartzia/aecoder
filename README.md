# Autoencoder para detección de casquillos defectuosos
La entrada de datos son casquillos recortados en imagenes de __x__ pixeles.

## Preprocesado
El preprocesado, dado el pequeño número de imágenes, consiste en girar, mover levemente, e introducir otra clase de conversiones a las imágenes.

## Entrenamiento
Se entrena solo con imágenes buenas (sin anomalías).

## Inferencia
Consiste en utilizar el autoencoder entero y generar imágenes donde se segmenta se muestra en función de la intensidad de los grises la diferencia entre la imagen de la entrada, y la imagen de la salida del autoencoder. Cuando se detecten anomalías, la imagen de la salida tendrá un mayor error (una mayor diferencia respecto a la imagen de la entrada).
A partir de esa imagen, deberíamos fijar unos umbrales a partir de los cuales se den las imágenes o regiones segmentadas como anómalas.

## Extracciones
Para depurar y entender mejor el funcionamiento de la arquitectura de la red, se pueden extraer 3 datos diferentes.

### Espacio latente
Se trata del espacio vectorial que se encuentra entre el codificador y el decodificador del autoencoder. Mediante estos datos, se pretende entender en modo de características codificadas como la red ha aprendido a mapear las imágenes de la entrada a ese nuevo espacio vectorial de características, y así tratar de encontrar minuciosamente las diferencias entre imágenes buenas y malas en ese espacio vectorial.

### Segmentos
Se tratan de las imágenes de diferencia entre entrada y salida del autoencoder. Es parecido a lo que se realiza en la inferencia, pero sin el umbral añadido.

### Errores
Se trata del error (elegido como error ya sea MSE o cualquier otro) total entre la salida y la entrada del autoencoder.

### Imágenes
Se trata de las imágenes de la salida del autoencoder, sin tratar, ni calcular la diferencia respecto a la entrada.



