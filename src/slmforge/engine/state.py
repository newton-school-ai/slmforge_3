from sqlalchemy import Column, Integer, String
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class Build(Base):
    __tablename__ = "builds"

    id = Column(Integer, primary_key=True)
    name = Column(String)
    build_id = Column(String, unique=True, index=True, nullable=True)
    checkpoint_path = Column(String, nullable=True)
    seed = Column(Integer, nullable=True)
    epochs_completed = Column(Integer, nullable=True)
    total_epochs = Column(Integer, nullable=True)
    status = Column(String, nullable=True)


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
