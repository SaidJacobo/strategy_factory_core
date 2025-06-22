from contextlib import contextmanager
from typing import Iterable
from sqlalchemy import UUID
from sqlalchemy.orm import Session
from app.backbone.database.db import get_db
from sqlalchemy.sql import and_

class DbService:
    def __init__(self):
        pass  # No necesita estar vinculado a un modelo específico

    def get_by_id(self, db: Session, model, id: int):
        return db.query(model).filter(model.Id == id).first()

    def get_by_filter(self, db: Session, model, **filters):
        return db.query(model).filter(and_(*[getattr(model, key) == value for key, value in filters.items()])).first()

    def get_many_by_filter(self, db: Session, model, **filters):
        return db.query(model).filter(and_(*[getattr(model, key) == value for key, value in filters.items()])).all()
    
    def delete_many_by_filter(self, db: Session, model, **filters):
        return db.query(model).filter(and_(*[getattr(model, key) == value for key, value in filters.items()])).delete()

    def get_all(self, db: Session, model):
        return db.query(model).all()

    def create(self, db: Session, obj_in):
        db.add(obj_in)
        return obj_in

    def create_all(self, db, iterable: Iterable[object]) -> None:
        db.add_all(iterable)
        
        return None

    def update(self, db: Session, model, obj_in):
        db_obj = db.query(model).filter(model.Id == obj_in.Id).first()
        db.merge(obj_in)
        return db_obj

    def delete(self, db: Session, model, id: UUID):
        obj = db.query(model).filter(model.Id == id).first()
        if obj:
            db.delete(obj)
        return obj
    
    def delete_all(self, db: Session, model):
        db.query(model).delete()

    def save(self, db: Session):
        try:
            db.commit()
        except Exception as e:
            db.rollback()
            raise e

    @contextmanager
    def get_database(self):
        with get_db() as db:
            try:
                db.expire_on_commit = False
                yield db
                db.commit()
            except Exception as e:
                db.rollback()
                raise e
