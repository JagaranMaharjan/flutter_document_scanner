// Copyright (c) 2021, Christian Betancourt
// https://github.com/criistian14
//
// Use of this source code is governed by an MIT-style
// license that can be found in the LICENSE file or at
// https://opensource.org/licenses/MIT.

import 'package:flutter/material.dart';
import 'package:flutter_document_scanner/src/models/area.dart';

class BorderCropAreaPainter extends CustomPainter {
  const BorderCropAreaPainter({
    required this.area,
    this.colorBorderArea,
    this.widthBorderArea,
  });

  final Area area;
  final Color? colorBorderArea;
  final double? widthBorderArea;

  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..style = PaintingStyle.stroke
      ..strokeWidth = widthBorderArea ?? 3
      ..color = colorBorderArea ?? Colors.white;

    final path = Path()
      ..moveTo(area.topLeft.x, area.topLeft.y)
      ..lineTo(area.topRight.x, area.topRight.y)
      ..lineTo(area.bottomRight.x, area.bottomRight.y)
      ..lineTo(area.bottomLeft.x, area.bottomLeft.y)
      ..close();

    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(CustomPainter oldDelegate) => true;
}
