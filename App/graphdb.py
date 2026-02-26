from typing import List
from neo4j import GraphDatabase

class Neo4jRetriever:
    def __init__(self, uri: str, user: str, password: str, database: str = "neo4j"):
        if GraphDatabase is None:
            raise RuntimeError("neo4j driver missing")

        self.driver = GraphDatabase.driver(
            uri,
            auth=(user, password)
        )
        self.database = database

    def get_candidates(
        self,
        anchor_text: str,
        source: str,
        limit: int = 5
    ) -> List[str]:  # type: ignore

        query = """
        MATCH (ent:Entity {source: $source})
        WHERE toLower($text) CONTAINS toLower(ent.name)
        OR toLower(ent.name) CONTAINS toLower($text)

        MATCH (ent)<-[:HAS_ACTOR|HAS_ACTION|HAS_OBJECT|HAS_ATTRIBUTE]-(r:Requirement {source: $source})
        WHERE r.text <> $text

        RETURN DISTINCT r.text AS candidate_text
        LIMIT $k
        """

        with self.driver.session(database=self.database) as session:
            res = session.run(
                query,
                text=anchor_text,
                source=source,
                k=limit
            )

            return [
                record["candidate_text"]
                for record in res
                if record["candidate_text"]
            ]

    def close(self):
        if self.driver:
            self.driver.close()