from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Build(Base):
    __tablename__ = "builds"

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Run(Base):
    __tablename__ = "runs"

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Source(Base):
    __tablename__ = "sources"

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Eval(Base):
    __tablename__ = "evals"

    id = Column(Integer, primary_key=True)
    name = Column(String)


class Serve(Base):
    __tablename__ = "serves"

    id = Column(Integer, primary_key=True)
    name = Column(String)
