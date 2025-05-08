Usar cross validation


Los modelos más sensibles a la escala y al ruido (como SVR y KNN) han mejorado significativamente cuando se han aplicado preprocesamientos adecuados:

SVR :

  requiere normalización, y por eso rinde mucho mejor que el árbol.

KNN:
  que se basa en distancias, mejora mucho tras eliminar outliers o limpiar datos ruidosos (de 10.5 a 7.3).

El árbol de decisión:

  No necesita normalización ni codificación especial.

  Es más robusto frente a datos sucios, pero no tan preciso. Te da una base decente, pero no es el mejor modelo si quieres precisión.

La red neuronal (MLP):

  Ha dado el mejor RMSE de todos, lo cual tiene sentido porque es capaz de capturar relaciones no lineales complejas si está bien entrenada.

  Este rendimiento sugiere que la relación entre los hábitos del estudiante y su nota no es lineal, y por eso un modelo más sofisticado como la red lo capta mejor.

Conclusión general:

  Los resultados tienen sentido porque reflejan el tipo de modelo y su sensibilidad al preprocesamiento:

  Modelos simples (como árboles o KNN sin limpieza) hacen predicciones razonables pero con mayor error.

  Modelos más complejos o bien preprocesados (SVR, redes, KNN limpio) capturan mejor el patrón de los datos.
