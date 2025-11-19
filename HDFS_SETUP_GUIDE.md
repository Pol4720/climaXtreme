# Guía de configuración de HDFS distribuido

Esta guía describe la arquitectura y configuración del sistema de archivos HDFS distribuido utilizado en el proyecto climaXtreme.

## Arquitectura

El clúster de HDFS está compuesto por los siguientes componentes:

*   **1 NameNode**: El NameNode es el nodo maestro que gestiona el espacio de nombres del sistema de archivos y regula el acceso a los archivos por parte de los clientes. Mantiene el árbol del sistema de archivos y los metadatos de todos los archivos y directorios del árbol.
*   **3 DataNodes**: Los DataNodes son los nodos de trabajo que almacenan los datos. Son responsables de servir las solicitudes de lectura y escritura de los clientes del sistema de archivos. También realizan la creación, eliminación y replicación de bloques por instrucción del NameNode.

## Configuración

La configuración del clúster de HDFS se define en los siguientes archivos:

*   `infra/docker-compose.yml`: Este archivo define los servicios que componen el clúster de Hadoop, incluyendo el NameNode y los tres DataNodes.
*   `infra/hadoop.env`: Este archivo contiene las variables de entorno para la configuración de Hadoop. La variable más importante para la arquitectura distribuida es `HDFS_CONF_dfs_replication`, que está configurada en `3`. Esto asegura que cada bloque de datos se replique en los tres DataNodes, proporcionando alta disponibilidad y tolerancia a fallos.

## Uso

Para iniciar el clúster de HDFS, ejecute el siguiente comando desde el directorio raíz del proyecto:

```bash
docker-compose -f infra/docker-compose.yml up -d
```

Una vez que el clúster esté en funcionamiento, la aplicación puede interactuar con HDFS a través del NameNode en la dirección `hdfs://climaxtreme-namenode:9000`.
