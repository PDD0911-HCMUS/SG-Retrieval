from typing import Optional

from sqlalchemy import ARRAY, BigInteger, Boolean, Date, Double, Identity, Integer, PrimaryKeyConstraint, Text, Uuid
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column
import datetime
import uuid

class Base(DeclarativeBase):
    pass


class GraphRetrieval(Base):
    __tablename__ = 'GraphRetrieval'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='GraphRetrieval_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name_qu: Mapped[Optional[str]] = mapped_column(Text)
    que_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    image_name_rev: Mapped[Optional[str]] = mapped_column(Text)
    rev_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))


class GraphRetrievalV2(Base):
    __tablename__ = 'GraphRetrieval_V2'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='GraphRetrieval_V2_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name_qu: Mapped[Optional[str]] = mapped_column(Text)
    que_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    image_name_rev: Mapped[Optional[str]] = mapped_column(Text)
    rev_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))


class GraphRetrievalV2MSCOCO(Base):
    __tablename__ = 'GraphRetrieval_V2_MSCOCO'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='GraphRetrieval_V2_MSCOCO_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name_qu: Mapped[Optional[str]] = mapped_column(Text)
    que_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    image_name_rev: Mapped[Optional[str]] = mapped_column(Text)
    rev_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))


class GraphRetrievalAllMiniLML6V2(Base):
    __tablename__ = 'GraphRetrieval_all-MiniLM-L6-v2'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='GraphRetrieval_all-MiniLM-L6-v2_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name_qu: Mapped[Optional[str]] = mapped_column(Text)
    que_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    image_name_rev: Mapped[Optional[str]] = mapped_column(Text)
    rev_embeding: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))


class Image2GraphEmbedding(Base):
    __tablename__ = 'Image2GraphEmbedding'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Image2GraphEmbedding_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name: Mapped[Optional[str]] = mapped_column(Text)
    embeding_value: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
    triplets: Mapped[Optional[str]] = mapped_column(Text)


class Image2GraphEmbeddingV2(Base):
    __tablename__ = 'Image2GraphEmbedding_V2'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Image2GraphEmbedding_V2_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name: Mapped[Optional[str]] = mapped_column(Text)
    triplets: Mapped[Optional[str]] = mapped_column(Text)


class Image2GraphEmbeddingV2MSCOCO(Base):
    __tablename__ = 'Image2GraphEmbedding_V2_MSCOCO'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Image2GraphEmbedding_V2_MSCOCO_pkey'),
    )

    ID: Mapped[int] = mapped_column(BigInteger, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=9223372036854775807, cycle=False, cache=1), primary_key=True)
    image_name: Mapped[Optional[str]] = mapped_column(Text)
    triplets: Mapped[Optional[str]] = mapped_column(Text)


class Pages(Base):
    __tablename__ = 'Pages'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='Pages_pkey'),
    )

    ID: Mapped[uuid.UUID] = mapped_column(Uuid, primary_key=True)
    PageName: Mapped[Optional[str]] = mapped_column(Text)
    PageURL: Mapped[Optional[str]] = mapped_column(Text)
    PageLogo: Mapped[Optional[str]] = mapped_column(Text)
    RoleID: Mapped[Optional[list]] = mapped_column(ARRAY(Integer()))
    Activate: Mapped[Optional[bool]] = mapped_column(Boolean)
    Delete: Mapped[Optional[bool]] = mapped_column(Boolean)
    CreateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    UpdateAt: Mapped[Optional[datetime.date]] = mapped_column(Date)
    DeleteAt: Mapped[Optional[datetime.date]] = mapped_column(Date)


class TestTable(Base):
    __tablename__ = 'TestTable'
    __table_args__ = (
        PrimaryKeyConstraint('ID', name='TestTable_pkey'),
    )

    ID: Mapped[int] = mapped_column(Integer, Identity(always=True, start=1, increment=1, minvalue=1, maxvalue=2147483647, cycle=False, cache=1), primary_key=True)
    TestContent1: Mapped[Optional[str]] = mapped_column(Text)
    TestContent2: Mapped[Optional[list]] = mapped_column(ARRAY(Double(precision=53)))
