class Lottery:
  def __init__(self, type, number, series,  year, month):
    self.type = type
    self.number = number
    self.series = series
    self.year = year
    self.month = month

  def to_dict(self):
      return {
        "type": self.type,
        "number": self.number,
        "series": self.series,
        "year": self.year,
        "month": self.month
      }

  @staticmethod
  def from_dict(data):
      return Lottery(
          type=data.get("type"),
          number=data.get("number"),
          series=data.get("series"),
          year=data.get("year"),
          month=data.get("month")
      )
  
  def exists_in_db(self, db):
        query = """
        SELECT 1 FROM lottery WHERE year = ? AND month = ? AND number = ? AND series = ?
        """
        result = db.get_by_id_custom(query, (self.year, self.month, self.number, self.series))
        return result is not None
      
  def save_to_db(self, db_controller):
      if self.exists_in_db(db_controller):
          return
      db_controller.insert("lottery", self.to_dict())

